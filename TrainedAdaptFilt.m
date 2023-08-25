%% load data from file 
load("sim4_5data.mat");
g = squeeze(noise_g.Data); 
d = squeeze(unfiltered_d.Data);
t = unfiltered_d.Time;

%% identify parameters for filter and training 
trainfrac = .5;
N = 256; % filter taps 
stepsize = 1e7;
nEpoch = 2500;

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

% iterate grad descent 
figure; subplot(211); wplot = stem(w); subplot(212); eplot = plot(e_t);
pause(.5);
for ep = 1:nEpoch
    E = D - G*w;
    e_t(ep) = mean(E);
    dW = E.*G;
    dw = mean(dW,1)';
    w = w + stepsize*dw;
    wplot.YData = w; eplot.YData = e_t;
    pause(eps);
end

% demo final signal 