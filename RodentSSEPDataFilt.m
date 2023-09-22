%% load file
TDTPATH = 'TDTMatlabSDK';
addpath(genpath(TDTPATH));
data = TDTbin2mat('Rodent SSEP Data/AC5-230830-130841');

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
%uchan = uchan(1); % comment out to get all chans
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
d         = filter(hpFilt, d_unfilt);
%% lowpass filtering (noise removal)
lpFilt = designfilt('lowpassiir', ...
                    'StopbandFrequency', 500, ...
                    'PassbandFrequency', 480, ...
                    'PassbandRipple', .5, ...
                    'StopbandAttenuation', 60, ...
                    'SampleRate', Fs, ... 
                    'DesignMethod', 'butter');
%fvtool(hpFilt);

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
%w_OL = zeros(N, length(uchan));
w_OL = w;
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
        E = d(ep+N-1,idx) - Gidx*w_OL(:,idx);
        e_t(ep, idx) = E;
        dw = E*Gidx';
        w_OL(:,idx) = w_OL(:,idx) + stepsize*dw;
        if ~mod(ep, floor(size(t,1)/nUpdates))
            wplot.YData = w_OL(:,idx); eplot.YData = movmean(e_t(:,idx).^2, 5000);
            disp(['Online Channel ',num2str(uchan(idx)),': ',num2str(100*ep/size(t,1)),'%'])
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
    plot(t(N:end,idx)-.007, e_t_lpf(:,idx)); % is alignment valid???
    grid on;
    xlabel('time (s)'); ylabel('filtered signal (V)');
%    legend('original', 'train', 'test', 'online');
    legend('original', 'adaptive filtered');
    title(['channel ',num2str(uchan(idx))])
    
    %xlim([1410.1, 1411.4])
    xlim([1410.351, 1410.449])
    %xlim([336.1, 337.1])
    ylim([-8e-5, 8e-5])
end

%ylim([-8e-5, 8e-5])
%xlim([336.351, 336.449])
%xlim([336.1, 337.1])

%% getting signals before and after stim  
tBeforeTrig = .29; % s
nBeforeTrig = floor(tBeforeTrig*Fs); % samples
tBeforeTrig = nBeforeTrig/Fs;
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

    d_PrePost_ch           = zeros(length(trig), nBeforeTrig+1, 2);
    d_lpf_PrePost_ch       = zeros(size(d_PrePost_ch));
    e_train_PrePost_ch     = zeros(size(d_PrePost_ch));
    e_train_lpf_PrePost_ch = zeros(size(d_PrePost_ch));
    e_test_PrePost_ch      = zeros(size(d_PrePost_ch));
    e_test_lpf_PrePost_ch  = zeros(size(d_PrePost_ch));
    e_t_PrePost_ch         = zeros(size(d_PrePost_ch));
    e_t_lpf_PrePost_ch     = zeros(size(d_PrePost_ch));

    for trIdx = 1:length(trig)
        tr = trig(trIdx); % timepoint

        d_PrePost_ch(trIdx,:,1) = d(tr + ((-nBeforeTrig):0), idx);
        d_PrePost_ch(trIdx,:,2) = d(tr + (  0:nBeforeTrig ), idx);
        d_lpf_PrePost_ch(trIdx,:,1) = d_lpf(tr + ((-nBeforeTrig):0), idx);
        d_lpf_PrePost_ch(trIdx,:,2) = d_lpf(tr + (  0:nBeforeTrig ), idx);
        e_t_PrePost_ch(trIdx,:,1) = e_t(tr + ((-nBeforeTrig):0), idx);
        e_t_PrePost_ch(trIdx,:,2) = e_t(tr + (  0:nBeforeTrig ), idx);
        e_t_lpf_PrePost_ch(trIdx,:,1) = e_t_lpf(tr + ((-nBeforeTrig):0), idx);
        e_t_lpf_PrePost_ch(trIdx,:,2) = e_t_lpf(tr + (  0:nBeforeTrig ), idx);
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
% colors: 
dkBlue  = [  1,  50, 130] /255;
ltBlue  = [145, 190, 255] /255;
dkRed   = [110,   0,   0] /255;
ltRed   = [250, 150, 150] /255;
dkBlack = [  0,   0,   0] /255;
ltBlack = [110, 110, 110] /255;

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

    [~, ~, wFiltBefore, spectFiltBeforeCh] = PowerSpectrum(sigFiltCh(:,:,1), Fs);
    [~, ~, wFiltAfter, spectFiltAfterCh] = PowerSpectrum(sigFiltCh(:,:,2), Fs);
    [~, ~, wUnfiltBefore, spectUnfiltBeforeCh] = PowerSpectrum(sigUnfiltCh(:,:,1), Fs);
    [~, ~, wUnfiltAfter, spectUnfiltAfterCh] = PowerSpectrum(sigUnfiltCh(:,:,2), Fs);

    spectFiltBeforeCh = abs(spectFiltBeforeCh);
    spectFiltAfterCh = abs(spectFiltAfterCh);
    spectUnfiltBeforeCh = abs(spectUnfiltBeforeCh); 
    spectUnfiltAfterCh = abs(spectUnfiltAfterCh);

    meanSpectFiltBefore = mean(spectFiltBeforeCh); 
    errbSpectFiltBefore =  std(spectFiltBeforeCh);
    meanSpectFiltAfter = mean(spectFiltAfterCh); 
    errbSpectFiltAfter =  std(spectFiltAfterCh);
    meanSpectUnfiltBefore = mean(spectUnfiltBeforeCh); 
    errbSpectUnfiltBefore =  std(spectUnfiltBeforeCh);
    meanSpectUnfiltAfter = mean(spectUnfiltAfterCh); 
    errbSpectUnfiltAfter =  std(spectUnfiltAfterCh);

    figure('Units','normalized', 'Position',[.1 .1 .8 .8]); 
    sgtitle(['Chennel ',num2str(uchan(idx)),' Avg. Response to Stim']);

    subplot(321); 
           plotWithDistrib(t_PrePost(1,:), meanUnfiltBefore, errbUnfiltBefore, ltRed);
    yrng = plotWithDistrib(t_PrePost(1,:), meanFiltBefore, errbFiltBefore, dkBlue);
    title('Filtered Before'); grid on; 
    xlabel('time (s)'); ylabel('Signal (V)'); 
    ylim(yrng(2,:)); 
    legend('Unfiltered', '-1SD', '+1SD', 'Filtered', '-1SD', '+1SD', 'Location','eastoutside');

    subplot(322); 
           plotWithDistrib(t_PrePost(2,:), meanUnfiltAfter, errbUnfiltAfter, ltRed);
    yrng = plotWithDistrib(t_PrePost(2,:), meanFiltAfter, errbFiltAfter, dkBlue);
    title('Filtered After'); grid on; 
    xlabel('time (s)'); ylabel('Signal (V)');
    ylim(yrng(2,:));
    legend('Unfiltered', '-1SD', '+1SD', 'Filtered', '-1SD', '+1SD', 'Location','eastoutside');

    subplot(323);  
           plotWithDistrib(t_PrePost(1,:), meanFiltBefore, errbFiltBefore, ltBlue);
    yrng = plotWithDistrib(t_PrePost(1,:), meanUnfiltBefore, errbUnfiltBefore, dkRed);
    title('Unfiltered Before'); grid on; 
    xlabel('time (s)'); ylabel('Signal (V)');
    ylim(yrng(2,:));
    legend('Filtered', '-1SD', '+1SD', 'Unfiltered', '-1SD', '+1SD', 'Location','eastoutside');

    subplot(324);  
           plotWithDistrib(t_PrePost(2,:), meanFiltAfter, errbFiltAfter, ltBlue);
    yrng = plotWithDistrib(t_PrePost(2,:), meanUnfiltAfter, errbUnfiltAfter, dkRed);
    title('Unfiltered After'); grid on; 
    xlabel('time (s)'); ylabel('Signal (V)');
    ylim(yrng(2,:));
    legend('Filtered', '-1SD', '+1SD', 'Unfiltered', '-1SD', '+1SD', 'Location','eastoutside');

    subplot(325); 
    semilogy(wUnfiltAfter, meanSpectUnfiltAfter, 'Color', ltRed); hold on; 
    semilogy(wFiltAfter, meanSpectFiltAfter, 'Color', ltBlue);
    plotWithDistrib(wUnfiltBefore, meanSpectUnfiltBefore, errbSpectUnfiltBefore, dkRed);
    plotWithDistrib(wFiltBefore, meanSpectFiltBefore, errbSpectFiltBefore, dkBlue);
    title('Spectrum Before'); grid on; 
    %set(gca, 'YScale', 'log');
    xlabel('Frequency (Hz)'); ylabel('Magnitude Spectrum (V*s)');
    legend('Unfiltered After', 'Filtered After', ...
        'Unfiltered Before', '-1SD', '+1SD', 'Filtered Before', '-1SD', '+1SD', 'Location','eastoutside');

    subplot(326); 
    semilogy(wUnfiltBefore, meanSpectUnfiltBefore, 'Color', ltRed); hold on; 
    semilogy(wFiltBefore, meanSpectFiltBefore, 'Color', ltBlue);
    plotWithDistrib(wUnfiltAfter, meanSpectUnfiltAfter, errbSpectUnfiltAfter, dkRed);
    plotWithDistrib(wFiltAfter, meanSpectFiltAfter, errbSpectFiltAfter, dkBlue);
    title('Spectrum After'); grid on; 
    %set(gca, 'YScale', 'log');
    xlabel('Frequency (Hz)'); ylabel('Magnitude Spectrum (V*s)');
    legend('Unfiltered Before', 'Filtered Before', ...
        'Unfiltered After', '-1SD', '+1SD', 'Filtered After', '-1SD', '+1SD', 'Location','eastoutside');
end

%% helper functions 
function range = plotWithDistrib(x, y, dist, colr)
    % plot y with a dashed +- distribution surrounding y. 
    % y and dist must be row vectors
    plot(x, y, 'Color', colr); 
    hold on; 
    Y = y + [1;-1].*dist;
    plot(x, Y, ':', 'Color', colr);
    range = [min(Y(:)), max(Y(:))];
    range = [range; 1.25*[-1,1]*.5*diff(range) + mean(range)];
end

function [wP, P, w, Y] = PowerSpectrum(y, Fs)
    % y: a row vector/matrix in time domain 
    % wP: one-sided frequency 
    % P: power spectrum (1-sided) of Y
    % w: two-sided frequency 
    % Y: frequency spectrum (2-sided, complex)
    L = size(y, 2);
    y = y';
    Y = fft(y);  
    w = Fs/L*(-L/2:L/2-1);
    P = abs(Y/L)'; P = P(:, 1:L/2+1);
    P(:, 2:end-1) = 2*P(:, 2:end-1);
    P = P.^2;
    wP = Fs/L*(0:(L/2));
    Y = fftshift(Y)';
end