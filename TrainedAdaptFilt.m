%% load data from file 
load("sim4_5data.mat");
g = squeeze(noise_g.Data); 
d_unfilt = squeeze(unfiltered_d.Data);
t = unfiltered_d.Time;
load('20210715_1048.mat_EyesOpen_Trial1.mat');
t_clean   = TimesAboveData(1,:);
EEG_clean_unfilt = TimesAboveData(2,:)*1e-6; % V
EEG_clean_unfilt = interp1(t_clean, EEG_clean_unfilt, t); %EEG_clean now corresponds to t

%% identify parameters for filter and training 
trainfrac = .5;
N = 128; % filter taps 
stepsize = 1e7;
nEpoch = 50000;
nUpdates = 100;

%% highpass filtering (baseline removal) 
hpFilt = designfilt('highpassiir', ...
                    'StopbandFrequency', .5, ...
                    'PassbandFrequency', 1.5, ...
                    'PassbandRipple', .5, ...
                    'StopbandAttenuation', 60, ...
                    'SampleRate', 1000, ...
                    'DesignMethod', 'butter');
%fvtool(hpFilt);
d         = filter(hpFilt, d_unfilt);
EEG_clean = filter(hpFilt, EEG_clean_unfilt);

%% run through 

% split testing and training 
splIdx = floor(trainfrac*length(t));
t_test = t(1:splIdx); t_train = t((splIdx+1):end);
g_test = g(1:splIdx); g_train = g((splIdx+1):end);
d_test = d(1:splIdx); d_train = d((splIdx+1):end);

% organize training epochs 
G = zeros(length(t_train)-N+1, N); 
T = zeros(size(G)); 
D = zeros(length(t_train)-N+1,1);
D(:) = d_train(N:length(t_train));
for nf = 1:(length(t_train)-N+1)
    G(nf,:) = g_train(nf:(nf+N-1));
    T(nf,:) = t_train(nf:(nf+N-1));
end
w = zeros(N,1);
e_t = zeros(1,nEpoch);

%{
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
w = (((G'*G)^-1)*G')*D;
figure; stem(w);

% organize testing epochs 
G_test = zeros(length(t_test)-N+1, N); 
T_test = zeros(size(G_test)); 
D_test = zeros(length(t_test)-N+1,1);
D_test(:) = d_test(N:length(t_test));
for nf = 1:(length(t_test)-N+1)
    G_test(nf,:) = g_test(nf:(nf+N-1));
    T_test(nf,:) = t_test(nf:(nf+N-1));
end

%% online LMS for comparison 
w_OL = zeros(N,1);
e_t = zeros(size(G,1),1);

% train w: iterate grad descent 
figure('Units','normalized', 'Position',[.1 .1 .8 .8]); 
subplot(211); wplot = stem(w_OL); grid on; 
subplot(212); eplot = semilogy(e_t); grid on;
pause(.5);
for ep = 1:size(G,1)
    E = D(ep) - G(ep,:)*w_OL;
    e_t(ep) = E;
    dw = E*G(ep,:)';
    w_OL = w_OL + stepsize*dw;
    if ~mod(ep, floor(size(G,1)/nUpdates))
        wplot.YData = w_OL; eplot.YData = e_t.^2;
        pause(eps);
    end
end

%% demo final signal 
op_train = G*w;
e_train = d_train; e_train(N:end) = e_train(N:end) - op_train;
op_test = G_test*w;
e_test = d_test; e_test(N:end) = e_test(N:end) - op_test;
figure; plot(t, EEG_clean, 'k', 'LineWidth', 1); hold on;
plot(t_train, e_train); plot(t_test, e_test);
plot(t_train(N:end), e_t); 
grid on;
xlabel('time (s)'); ylabel('filtered signal (V)'); 
legend('original', 'train', 'test', 'online');