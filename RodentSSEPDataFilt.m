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
trainfrac = .2;
N = 128; % filter taps 
stepsize = 1e7;
nEpoch = 50000;
nUpdates = 100;
maxBlockSize = 3e6; % time points 

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

%% organize into testing and training epochs

% split testing and training 
splIdx = floor(trainfrac*size(t,1));
t_train = t(1:splIdx, :); t_test = t((splIdx+1):end, :);
g_train = g(1:splIdx, :); g_test = g((splIdx+1):end, :);
d_train = d(1:splIdx, :); d_test = d((splIdx+1):end, :);
% reduce training size to fit within max block size

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

% organize testing epochs 
G_test = zeros(size(t_test,1)-N+1, N, length(uchan)); 
T_test = zeros(size(G_test)); 
D_test = zeros(size(t_test,1)-N+1, length(uchan));
for idx = 1:length(uchan)
    D_test(:,idx) = d_test(N:size(t_test,1), idx);
    for nf = 1:(size(t_train,1)-N+1)
        G_test(nf,:,idx) = g_test(nf:(nf+N-1), idx);
        T_test(nf,:,idx) = t_test(nf:(nf+N-1), idx);
    end
end

%% run through 

%{
w = zeros(N,1);
e_t = zeros(1,nEpoch);

% train w: iterate grad descent 
figure('Units','normalized', 'Position',[.1 .1 .8 .8]); 
subplot(211); wplot = errorbar(w, w, 'x'); grid on; 
subplot(212); eplot = semilogy(e_t); grid on;
pause(.5);
for ep = 1:nEpoch
    E = D - G*w;
    e_t(ep) = mean(E.^2);
    dW = E.*G;
    dw = mean(dW,1)';
    w = w + stepsize*dw;
    if ~mod(ep, floor(nEpoch/nUpdates))
        wplot.YData = w; eplot.YData = e_t;
        wplot.YNegativeDelta = stepsize*std(dW, [], 1)/2; 
        wplot.YPositiveDelta = stepsize*std(dW, [], 1)/2;
        pause(eps);
    end
end
%}

w = zeros(N, length(uchan));
for idx = 1:length(uchan)
    Gidx = G(:,:,idx); Didx = D(:,idx);
    w(:,idx) = (((Gidx'*Gidx)^-1)*Gidx')*Didx;
end
figure; stem(w);

%% online LMS for comparison 
w_OL = zeros(N, length(uchan));
e_t = zeros(size(G,1), length(uchan));

for idx = 1:length(uchan)
    % train w: iterate grad descent
    Gidx = G(:,:,idx); Didx = D(:,idx);
    figure('Units','normalized', 'Position',[.1 .1 .8 .8]);
    subplot(211); wplot = stem(w_OL(:,idx)); grid on;
    subplot(212); eplot = semilogy(e_t(:,idx)); grid on;
    pause(.5);
    for ep = 2:size(Gidx,1)
        E = Didx(ep) - Gidx(ep,:)*w_OL(:,idx);
        e_t(ep, idx) = E;
        dw = E*Gidx(ep,:)';
        w_OL(:,idx) = w_OL(:,idx) + stepsize*dw;
        if ~mod(ep, floor(size(Gidx,1)/nUpdates))
            wplot.YData = w_OL(:,idx); eplot.YData = e_t(:,idx).^2;
            pause(eps);
        end
    end
end

%% post-processing and filtering 
op_train = zeros([size(t_train,1)-N+1,size(t_train,2)]); 
op_test  = zeros([size(t_test,1) -N+1,size(t_test, 2)]);
for idx = 1:length(uchan)
    op_train(:,idx) = G(:,:,idx)     *w(:,idx);
    op_test(:,idx)  = G_test(:,:,idx)*w(:,idx);
end
e_train = d_train; e_train(N:end,:) = e_train(N:end,:) - op_train;
e_test = d_test; e_test(N:end,:) = e_test(N:end,:) - op_test;

e_train = filter(lpFilt, e_train);
e_test  = filter(lpFilt, e_test);
e_t     = filter(lpFilt, e_t);
d       = filter(lpFilt, d);

%% demo final signal 
figure; 
for idx = 1:length(uchan)
    plot(t(:,idx), d(:,idx), 'k', 'LineWidth', 1); hold on;
    plot(t_train(:,idx), e_train(:,idx)); plot(t_test(:,idx), e_test(:,idx));
    plot(t_train(N:end,idx), e_t(:,idx));
    grid on;
    xlabel('time (s)'); ylabel('filtered signal (V)');
    legend('original', 'train', 'test', 'online');
    title(['channel ',num2str(uchan(idx))])
end