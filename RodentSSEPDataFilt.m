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
stepsize = .1;
nUpdates = 100;

%% "linearize" trial blocks 
t        = zeros(size(T,1)  *size(T,2),   length(uchan));
g        = zeros(size(G,1)  *size(G,2),   length(uchan));
d_unfilt = zeros(size(dta,1)*size(dta,2), length(uchan));
for idx = 1:length(uchan)
    Tidx = T(:,:,idx)'; Gidx = G(:,:,idx)'; Didx = dta(:,:,idx)';
    t(:,idx)        = Tidx(:);
    g(:,idx)        = Gidx(:);
    d_unfilt(:,idx) = Didx(:);
end

%% cleanup 
clear T G dta dta_t_chan
clear g_trl t_trl t_stim chan ch chIdx 

%% pre-filtering
% highpass filtering (baseline removal) 
hpFilt = designfilt('highpassiir', ...
                    'StopbandFrequency', .5, ...
                    'PassbandFrequency', 1.5, ...
                    'PassbandRipple', .5, ...
                    'StopbandAttenuation', 60, ...
                    'SampleRate', 1000, ...
                    'DesignMethod', 'butter');
% lowpass filtering (noise removal)
lpFilt = designfilt('lowpassiir', ...
                    'StopbandFrequency', 40, ...
                    'PassbandFrequency', 38, ...
                    'PassbandRipple', .5, ...
                    'StopbandAttenuation', 60, ...
                    'SampleRate', 1000, ...
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
            wplot.YData = w_OL(:,idx); eplot.YData = movmean(e_t(:,idx).^2, 3000);
            pause(eps);
        end
    end
end

%% post-processing and filtering 
op_train = zeros([size(t_train,1)-N+1,size(t_train,2)]); 
for idx = 1:length(uchan)
    op_train(:,idx) = G(:,:,idx)     *w(:,idx);
end

e_train = d_train; e_train(N:end,:) = e_train(N:end,:) - op_train;
e_test = d_test; e_test(N:end,:) = e_test(N:end,:) - op_test;

e_train_lpf = filter(lpFilt, e_train);
e_test_lpf  = filter(lpFilt, e_test);
e_t_lpf     = filter(lpFilt, e_t);
d_lpf       = filter(lpFilt, d);

%% demo final signal 
for idx = 1:length(uchan)
    figure; 
    plot(t(:,idx), d_lpf(:,idx), 'k', 'LineWidth', 1); hold on;
    plot(t_train(:,idx), e_train_lpf(:,idx)); plot(t_test(:,idx), e_test_lpf(:,idx));
    plot(t(N:end,idx), e_t_lpf(:,idx));
    grid on;
    xlabel('time (s)'); ylabel('filtered signal (V)');
    legend('original', 'train', 'test', 'online');
    title(['channel ',num2str(uchan(idx))])
end