x = data;
Len_1 = length(x);
X = pinknoise(Len_1)/10;
x = x + X;

wavename='db16'; % Debauchy Mother wavelet
level = 50; % Level of decomposition.
[cA,cD]= dwt(x,wavename); % Discrete wavelet transform
App=idwt(cA,[],wavename,Len_1); % Inverse discrete wavelet transform of signal to achieve approximate coefficient. 
Det=idwt([],cD,wavename,Len_1); % Inverse discrete wavelet transform of signal to achieve detailed coefficient.

[c,l]=wavedec(x,level,wavename); % Decomposition of corrupted signal.
cA3 = appcoef(c,l,wavename,level);% coefficient.
[cD1,cD2,cD3] = detcoef(c,l,[1,2,3]);% Detailed coefficient.

[App1,Det1] = dwt(x,wavename); % Discrete wavelet transform of noisy signal.
noise_level = median (abs(Det1))/0.6745; % Noise level Evaluation 
Threshold=sqrt(2*log(length(x)))*noise_level;
[thr,sorh,keepapp] = ddencmp('den','wv',x); % Default value for denoising.
sig_denoise=wdencmp('gbl',x,wavename,level,Threshold,sorh,keepapp); %One dimensional wavelet denoising by Global Thresholding.

  % Resultant Signal-to-Noise Ratio values of each stages
SNR_NSIG=snr (x); % SNR of contaminated signal
SNR_NSIG
DENOISED_SNR=snr (sig_denoise); %SNR of denoised Threshold Signal
DENOISED_SNR
FINAL_SNR=DENOISED_SNR-SNR_NSIG; %SNR of Reconstructed signal
FINAL_SNR

%%
figure;
plot(x,'r')
xlabel('Time (s)')
title('Original Signal')
figure();
plot(sig_denoise,'b')
xlabel('Time (s)')
title('Signal after denoising')