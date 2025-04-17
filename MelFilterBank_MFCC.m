%%SPECTROGRAM OF THE INPUT SPEECH
audioIn = data;

S = melSpectrogram(audioIn,fs);

[numBands,numFrames] = size(S);
fprintf("Number of bandpass filters in filterbank: %d\n",numBands)
fprintf("Number of frames in spectrogram: %d\n",numFrames)
figure();
melSpectrogram(audioIn,fs)
title('Mel Spectrogram')

%% Mel Spectrums of 2048-Point Windows of the Input Speech
S1 = melSpectrogram(audioIn,fs, ...
                   'Window',hamming(2048,'periodic'), ...
                   'OverlapLength',1024, ...
                   'FFTLength',4096, ...
                   'NumBands',64, ...
                   'FrequencyRange',[200,3e3]);
figure();
melSpectrogram(audioIn,fs, ...
               'Window',hamming(2048,'periodic'), ...
               'OverlapLength',1024, ...
               'FFTLength',4096, ...
               'NumBands',64, ...
               'FrequencyRange',[200,3e3])
title('Mel Spectrums of 2048-Point Windows')

%% Design Mel-Based Auditory Filter Bank

win = hamming(1024,"periodic");
noverlap = 512;
fftLength = 1024;
[S,F,t] = stft(audioIn,fs, ...
               "Window",win, ...
               "OverlapLength",noverlap, ...
               "FFTLength",fftLength, ...
               "FrequencyRange","onesided");
PowerSpectrum = S.*conj(S);

numBands = 32;
range = [0,4000];
normalization = "bandwidth";

[fb,cf] = designAuditoryFilterBank(fs, ...
                                   "FFTLength",fftLength, ...
                                   "NumBands",numBands, ...
                                   "FrequencyRange",range, ...
                                   "Normalization",normalization);

figure();
plot(F,fb.')
grid on
xlim([0 4000])
title("Mel Filter Bank")
xlabel("Frequency (Hz)")

X = fb*PowerSpectrum;

XdB = 10*log10(X);

figure();
surf(t,cf,XdB,"EdgeColor","none");
xlabel("Time (s)")
ylabel("Frequency (Hz)")
zlabel("Power (dB)")
view([45,60])
title('Mel-Based Spectrogram')
axis tight


freq1 = [];
freq2 = [];


maxim = [];
fm = [];


for k = 1:length(XdB)

    X = XdB(:,k);
    [pks,locs] = findpeaks(X,cf);
%     [maxi, maxIndex] = max(X);
%     maxim = [maxim maxi];
%     [row, col] = ind2sub(size(X), maxIndex);
   
end

for k = 1:length(locs)
    if(locs(k) > 300 && locs(k) < 800)
        freq1 = [freq1 locs(k)];
    end
    if(locs(k) > 800 && locs(k) < 3000 && length(freq2) < length(freq1) )
        freq2 = [freq2 locs(k)];
    end
end


%% Design Bark-Based Auditory Filter Bank

win = hamming(round(0.03*fs),"periodic");
noverlap = round(0.02*fs);
fftLength = 2048;

[S,F,t] = stft(audioIn,fs, ...
               "Window",win, ...
               "OverlapLength",noverlap, ...
               "FFTLength",fftLength, ...
               "FrequencyRange","onesided");
PowerSpectrum = S.*conj(S);

numBands = 32;
range = [0,22050];
normalization = "area";
designDomain = "linear";

[fb,cf] = designAuditoryFilterBank(fs, ...
    "FrequencyScale","bark", ...
    "FFTLength",fftLength, ...
    "NumBands",numBands, ...
    "FrequencyRange",range, ...
    "Normalization",normalization, ...
    "FilterBankDesignDomain",designDomain);

figure();
plot(F,fb.');
grid on
title("Bark Filter Bank")
xlabel("Frequency (Hz)")

X = pagemtimes(fb,PowerSpectrum);

XdB = 10*log10(X);

figure();
surf(t,cf,XdB,"EdgeColor","none");
xlabel("Time (s)")
ylabel("Frequency (Hz)")
view([0,90])
title("Barked-Based Spectrogram")
axis tight