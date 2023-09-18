%% load file
TDTPATH = 'TDTMatlabSDK';
addpath(genpath(TDTPATH));
data = TDTbin2mat('Rodent SSEP Data/AC5-230830-130841/AC5-230830-130841');

%% define parameters of stim 
% monophasic, fixed
%  ^         __________
%  |        |          |
% AmpA      |          |
%  |        |          |
%  v   _____|          |_____
%            <--DurA-->
PeriodA = 0.8;  % ms - between pulses?
CountA  = 1;    % pulse count? 
AmpA    = 1500; % uA
DurA    = 0.2;  % ms
DelayA  = 0;    % ms
ChanA   = 1; 

%% organize data from file
dta = data.snips.SSEP.data; 
t_stim = data.snips.SSEP.ts;
fs = data.snips.SSEP.fs;
chan = data.snips.SSEP.chan; uchan = unique(chan);

dta_t_chan = cell(2, length(uchan));
for idx = 1:length(uchan)
    ch = uchan(idx);
    chIdx = chan == ch;
    dta_t_chan{1, idx} =  dta(chIdx,:);
    dta_t_chan{2, idx} = t_stim(chIdx);
end

dta = zeros([size(dta_t_chan{1,1}), length(uchan)]);
t_stim = zeros([length(dta_t_chan{2,1}), length(uchan)]);
for idx = 1:length(uchan)
    dta(:,:,idx)  = dta_t_chan{1, idx};
    t_stim(:,idx) = dta_t_chan{2, idx};
end

t_trl = (1:size(dta, 2))/fs - .3; % ~ -.3 to +1 s
g_trl = (AmpA/1000)*((t_trl >= 0)&(t_trl < DurA/1000)); % noise reference, mA
G = repmat(g_trl, size(dta,1), 1, size(dta,3));

T = zeros(size(dta));
for idx = 1:length(uchan)
    T(:,:,idx) = t_stim(:,idx) + t_trl;
end

%% define parameters for filter and training 
trainfrac = .1;
N = 128; % filter taps 
stepsize = .2;
nUpdates = 100;

%% "linearize" trial blocks 
uchan = uchan(1); % comment out to get all chans
t        = zeros(size(T,1)  *size(T,2),   length(uchan));
g        = zeros(size(G,1)  *size(G,2),   length(uchan));
d_unfilt = zeros(size(dta,1)*size(dta,2), length(uchan));
for idx = 1:length(uchan)
    Tidx = T(:,:,idx)'; Gidx = G(:,:,idx)'; Didx = dta(:,:,idx)';

    % ensure correct order of timepoints 
    [Tidx, ord] = sort(Tidx(:));

    t(ord,idx)        = Tidx(:);
    g(ord,idx)        = Gidx(:);
    d_unfilt(ord,idx) = Didx(:);
end

% detect and fix inconsistencies in sampling 
% t must be in columns!
Dt = diff(t);
dt_mean = mean(Dt(:));
dt_min  = min(Dt(:));
dt_err  = std(Dt(:));
tLen = t(end,:) - t(1,:);
if dt_err > .01*dt_mean
    warning(['Inconsistent time steps; resampling at ',num2str(1/dt_min),' Hz']);
    t_from_start = 0:dt_min:max(tLen);

    t2 = zeros(length(t_from_start), length(uchan));
    g2 = zeros(length(t_from_start), length(uchan));
    d2 = zeros(length(t_from_start), length(uchan));

    for idx = 1:length(uchan)
        if tLen(idx) < max(tLen)
            warning(['Channel ',num2str(uchan(idx)),' has shorter duration and may be end-padded']);
        end
        t_ch = t_from_start + t(1,idx);

        t2(:,idx) = t_ch;
        g2(:,idx) = interp1(t(:,idx),        g(:,idx), t_ch, 'nearest','extrap');
        d2(:,idx) = interp1(t(:,idx), d_unfilt(:,idx), t_ch, 'nearest','extrap');
    end
    t        = t2; 
    g        = g2; 
    d_unfilt = d2; 

    dt_mean = dt_min;
end

Fs = 1/dt_mean; % Hz

%% cleanup 
clear Dt dt_mean dt_min dt_err tLen 
clear t2 g2 d2
clear T G dta dta_t_chan
clear g_trl t_trl t_stim chan ch chIdx 

%% pre-filtering
% highpass filtering (baseline removal) 
hpFilt = designfilt('highpassiir', ...
                    'StopbandFrequency', .5, ...
                    'PassbandFrequency', 1.5, ...
                    'PassbandRipple', .5, ...
                    'StopbandAttenuation', 60, ...
                    'SampleRate', Fs, ... 
                    'DesignMethod', 'butter');
% lowpass filtering (noise removal)
lpFilt = designfilt('lowpassiir', ...
                    'StopbandFrequency', 2000, ...
                    'PassbandFrequency', 1900, ...
                    'PassbandRipple', .5, ...
                    'StopbandAttenuation', 60, ...
                    'SampleRate', Fs, ... 
                    'DesignMethod', 'butter');
%fvtool(hpFilt);
d         = filter(hpFilt, d_unfilt);

%% organize into testing and training 

% split testing and training 
splIdx = floor(trainfrac*size(t,1));
t_train = t(1:splIdx, :); t_test = t((splIdx+1):end, :);
g_train = g(1:splIdx, :); g_test = g((splIdx+1):end, :);
d_train = d(1:splIdx, :); d_test = d((splIdx+1):end, :);

% organize training epochs 
G = zeros(size(t_train,1)-N+1, N, length(uchan)); 
T = zeros(size(G)); 
D = zeros(size(t_train,1)-N+1, length(uchan));
for idx = 1:length(uchan)
    D(:,idx) = d_train(N:size(t_train,1), idx);
    for nf = 1:(size(t_train,1)-N+1)
        G(nf,:,idx) = g_train(nf:(nf+N-1), idx);
        T(nf,:,idx) = t_train(nf:(nf+N-1), idx);
    end
end

%% training  
figure('Units','normalized', 'Position',[.1 .1 .4 .8]);
w = zeros(N, length(uchan));
for idx = 1:length(uchan)
    Gidx = G(:,:,idx); Didx = D(:,idx);
    w(:,idx) = (((Gidx'*Gidx)^-1)*Gidx')*Didx;
    subplot(length(uchan), 1, idx); stem(w(:,idx)); grid on;
    title(['Channel ',num2str(uchan(idx)),' training']);
    xlabel('tap'); ylabel('weight'); 
    pause(eps);
end
pause(.5);

%% testing  
op_test = zeros(size(t_test,1)-N+1, length(uchan));
for idx = 1:length(uchan)
    for ep = (N:size(t_test,1))-N+1 
        if ~mod(ep, floor(size(t_test,1)/(.1*nUpdates)))
            disp(['Testing Channel ',num2str(uchan(idx)),': ',num2str(100*ep/size(t_test,1)),'%']);
        end
        Gidx = g_test((1:N)+ep-1, idx)';
        op_test(ep,idx) = Gidx*w(:,idx);
    end
end

%% online LMS 
figure('Units','normalized', 'Position',[.1 .1 .8 .8]);
w_OL = zeros(N, length(uchan));
e_t = nan(size(t,1)-N+1, length(uchan));

for idx = 1:length(uchan)
    % train w: iterate grad descent
    subplot(length(uchan),2,2*idx-1); wplot = stem(w_OL(:,idx));    grid on;
    title(['Channel ',num2str(uchan(idx)),' online']);
    xlabel('tap'); ylabel('weight');
    subplot(length(uchan),2,2*idx);   eplot = semilogy(e_t(:,idx)); grid on;
    title(['Channel ',num2str(uchan(idx)),' online']);
    xlabel('timepoint'); ylabel('e^2');
    pause(.5);
    for ep = (N:size(t,1))-N+1
        Gidx = g((1:N)+ep-1, idx)';
        E = d(ep+N-1) - Gidx*w_OL(:,idx);
        e_t(ep, idx) = E;
        dw = E*Gidx';
        w_OL(:,idx) = w_OL(:,idx) + stepsize*dw;
        if ~mod(ep, floor(size(t,1)/nUpdates))
            wplot.YData = w_OL(:,idx); eplot.YData = movmean(e_t(:,idx).^2, 5000);
            disp(['Online Channel ',num2str(uchan(idx)),': ',num2str(ep/size(t,1)),'%'])
            pause(eps);
        end
    end
end

%% post-processing  
op_train = zeros([size(t_train,1)-N+1,size(t_train,2)]); 
for idx = 1:length(uchan)
    op_train(:,idx) = G(:,:,idx)     *w(:,idx);
end

e_train = d_train; e_train(N:end,:) = e_train(N:end,:) - op_train;
e_test = d_test; e_test(N:end,:) = e_test(N:end,:) - op_test;

%% post-filtering
disp('LP Filtering Train Signal')
e_train_lpf = filter(lpFilt, e_train);
disp('LP Filtering Test Signal')
e_test_lpf  = filter(lpFilt, e_test);
disp('LP Filtering Online Signal')
e_t_lpf     = filter(lpFilt, e_t);
disp('LP Filtering Original Signal')
d_lpf       = filter(lpFilt, d);

%% demo final signal 
for idx = 1:length(uchan)
    figure; 
    plot(t(:,idx), d(:,idx), 'k', 'LineWidth', 1); hold on;
%    plot(t_train(:,idx), e_train_lpf(:,idx)); plot(t_test(:,idx), e_test_lpf(:,idx));
    plot(t(N:end,idx)-.007, e_t_lpf(:,idx));
    grid on;
    xlabel('time (s)'); ylabel('filtered signal (V)');
%    legend('original', 'train', 'test', 'online');
    legend('original', 'adaptive filtered');
    title(['channel ',num2str(uchan(idx))])
end

ylim([-8e-5, 8e-5])
xlim([336.351, 336.449])

%% getting signals before and after stim  
tBeforeTrig = .29; % s
nBeforeTrig = tBeforeTrig*Fs; % samples
t_PrePost = [-tBeforeTrig:(1/Fs):0; 0:(1/Fs):tBeforeTrig]; % [before; after]

d_PrePost           = cell(1, length(uchan));
d_lpf_PrePost       = cell(size(d_PrePost));
e_train_PrePost     = cell(size(d_PrePost));
e_train_lpf_PrePost = cell(size(d_PrePost));
e_test_PrePost      = cell(size(d_PrePost));
e_test_lpf_PrePost  = cell(size(d_PrePost));
e_t_PrePost         = cell(size(d_PrePost));
e_t_lpf_PrePost     = cell(size(d_PrePost));

for idx = 1:length(uchan)
    gch = g(:,idx);
    trig = [0; abs(diff(gch))];
    trig = trig > .1*max(trig); trig = find(trig);

    d_PrePost_ch           = zeros(length(trig), nBeforeTrig, 2);
    d_lpf_PrePost_ch       = zeros(size(d_PrePost_ch));
    e_train_PrePost_ch     = zeros(size(d_PrePost_ch));
    e_train_lpf_PrePost_ch = zeros(size(d_PrePost_ch));
    e_test_PrePost_ch      = zeros(size(d_PrePost_ch));
    e_test_lpf_PrePost_ch  = zeros(size(d_PrePost_ch));
    e_t_PrePost_ch         = zeros(size(d_PrePost_ch));
    e_t_lpf_PrePost_ch     = zeros(size(d_PrePost_ch));

    for trIdx = 1:length(trig)
        tr = trig(trIdx); % timepoint

        d_PrePost_ch(trIdx,:,1) = d(tr + (-nBeforeTrig):0);
        d_PrePost_ch(trIdx,:,2) = d(tr +   0:nBeforeTrig );
        d_lpf_PrePost_ch(trIdx,:,1) = d_lpf(tr + (-nBeforeTrig):0);
        d_lpf_PrePost_ch(trIdx,:,2) = d_lpf(tr +   0:nBeforeTrig );
        e_train_PrePost_ch(trIdx,:,1) = e_train(tr + (-nBeforeTrig):0);
        e_train_PrePost_ch(trIdx,:,2) = e_train(tr +   0:nBeforeTrig );
        e_train_lpf_PrePost_ch(trIdx,:,1) = e_train_lpf(tr + (-nBeforeTrig):0);
        e_train_lpf_PrePost_ch(trIdx,:,2) = e_train_lpf(tr +   0:nBeforeTrig );
        e_test_PrePost_ch(trIdx,:,1) = e_test(tr + (-nBeforeTrig):0);
        e_test_PrePost_ch(trIdx,:,2) = e_test(tr +   0:nBeforeTrig );
        e_test_lpf_PrePost_ch(trIdx,:,1) = e_test_lpf(tr + (-nBeforeTrig):0);
        e_test_lpf_PrePost_ch(trIdx,:,2) = e_test_lpf(tr +   0:nBeforeTrig );
        e_t_PrePost_ch(trIdx,:,1) = e_t(tr + (-nBeforeTrig):0);
        e_t_PrePost_ch(trIdx,:,2) = e_t(tr +   0:nBeforeTrig );
        e_t_lpf_PrePost_ch(trIdx,:,1) = e_t_lpf(tr + (-nBeforeTrig):0);
        e_t_lpf_PrePost_ch(trIdx,:,2) = e_t_lpf(tr +   0:nBeforeTrig );
    end

    d_PrePost{idx} = d_PrePost_ch;
    d_lpf_PrePost{idx} = d_lpf_PrePost_ch;
    e_train_PrePost{idx} = e_train_PrePost_ch;
    e_train_lpf_PrePost{idx} = e_train_lpf_PrePost_ch;
    e_test_PrePost{idx} = e_test_PrePost_ch;
    e_test_lpf_PrePost{idx} = e_test_lpf_PrePost_ch;
    e_t_PrePost{idx} = e_t_PrePost_ch;
    e_t_lpf_PrePost{idx} = e_t_lpf_PrePost_ch;
end

%% cleanup 
clear d_PrePost_ch d_lpf_PrePost_che_train_PrePost_ch e_train_lpf_PrePost_ch 
clear e_test_PrePost_ch e_test_lpf_PrePost_ch e_t_PrePost_ch e_t_lpf_PrePost_ch 
clear gch trig trIdx tr

%% plotting averaged signals before and after stim 
for idx = 1:length(uchan)
    sigFiltCh = e_t_lpf_PrePost{idx};
    sigUnfiltCh = d_PrePost{idx};

    meanFiltBefore = mean(sigFiltCh(:,:,1));
    meanFiltAfter  = mean(sigFiltCh(:,:,2));
    errbFiltBefore =  std(sigFiltCh(:,:,1));
    errbFiltAfter  =  std(sigFiltCh(:,:,2));
    meanUnfiltBefore = mean(sigUnfiltCh(:,:,1));
    meanUnfiltAfter  = mean(sigUnfiltCh(:,:,2));
    errbUnfiltBefore =  std(sigUnfiltCh(:,:,1));
    errbUnfiltAfter  =  std(sigUnfiltCh(:,:,2));

    figure; 
    subplot(221); plot(t_PrePost(1,:), meanFiltBefore); title('Filtered Before')
    grid on; hold on; 
    plot(t_PrePost(1,:), meanFiltBefore + [1;-1].*[errbFiltBefore; errbFiltBefore], '--');
    subplot(222); plot(t_PrePost(2,:), meanFiltAfter); title('Filtered After')
    grid on; hold on; 
    plot(t_PrePost(2,:), meanFiltAfter  + [1;-1].*[errbFiltAfter ; errbFiltAfter ], '--');
    subplot(223); plot(t_PrePost(1,:), meanUnfiltBefore); title('Unfiltered Before')
    grid on; hold on; 
    plot(t_PrePost(1,:), meanUnfiltBefore + [1;-1].*[errbUnfiltBefore; errbUnfiltBefore], '--');
    subplot(224); plot(t_PrePost(2,:), meanUnfiltAfter); title('Unfiltered After')
    grid on; hold on; 
    plot(t_PrePost(2,:), meanUnfiltAfter  + [1;-1].*[errbUnfiltAfter ; errbUnfiltAfter ], '--');
end