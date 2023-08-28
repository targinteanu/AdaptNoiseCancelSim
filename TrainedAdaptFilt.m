%% load data from file 
load("sim4_5data.mat");
g = squeeze(noise_g.Data); 
d = squeeze(unfiltered_d.Data);
t = unfiltered_d.Time;
load('20210715_1048.mat_EyesOpen_Trial1.mat');
t_clean   = TimesAboveData(1,:);
EEG_clean = TimesAboveData(2,:)*1e-6; % V

%% identify parameters for filter and training 
trainfrac = .5;
N = 256; % filter taps 
stepsize = 1e7;
nEpoch = 2500;

%% highpass filtering (baseline removal) 
hpFilt = designfilt('highpassfir', ...
                    'StopbandFrequency', .5, ...
                    'PassbandFrequency', 1.5, ...
                    'PassbandRipple', .5, ...
                    'StopbandAttenuation', 60, ...
                    'SampleRate', 500, ...
                    'DesignMethod', 'equiripple');
%fvtool(hpFilt);
d         = filter(hpFilt, d);
EEG_clean = filter(hpFilt, EEG_clean);

%% run through 

% split testing and training 
splIdx = floor(trainfrac*length(t));
t_train = t(1:splIdx); t_test = t((splIdx+1):end);
g_train = g(1:splIdx); g_test = g((splIdx+1):end);
d_train = d(1:splIdx); d_test = d((splIdx+1):end);

% organize training epochs 
G = zeros(length(t_train)-N+1, N); 
T = zeros(size(G)); 
D = zeros(length(t_train)-N+1,1);
D(:) = d_train(N:length(t_train));
for nf = 1:(length(t_train)-N+1)
    G(nf,:) = g_train(nf:(nf+N-1));
    T(nf,:) = t_train(nf:(nf+N-1));
end
%W = zeros(size(G));
w = zeros(N,1);
e_t = zeros(1,nEpoch);

% train w: iterate grad descent 
figure('Units','normalized', 'Position',[.1 .1 .8 .8]); 
subplot(211); wplot = stem(w); grid on; 
subplot(212); eplot = semilogy(e_t); grid on;
pause(.5);
for ep = 1:nEpoch
    E = D - G*w;
    e_t(ep) = mean(E.^2);
    dW = E.*G;
    dw = mean(dW,1)';
    w = w + stepsize*dw;
    wplot.YData = w; eplot.YData = e_t;
    pause(eps);
end

% organize testing epochs 
G_test = zeros(length(t_test)-N+1, N); 
T_test = zeros(size(G_test)); 
D_test = zeros(length(t_test)-N+1,1);
D_test(:) = d_test(N:length(t_test));
for nf = 1:(length(t_test)-N+1)
    G_test(nf,:) = g_test(nf:(nf+N-1));
    T_test(nf,:) = t_test(nf:(nf+N-1));
end

%% demo final signal 
op_train = G*w;
e_train = d_train; e_train(N:end) = e_train(N:end) - op_train;
op_test = G_test*w;
e_test = d_test; e_test(N:end) = e_test(N:end) - op_test;
figure; plot(t_clean, EEG_clean, 'k', 'LineWidth', 1); 
hold on; plot(t_train, e_train); plot(t_test, e_test);
grid on;
xlabel('time (s)'); ylabel('filtered signal (V)'); legend('original', 'train', 'test');