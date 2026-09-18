function [rectified_signal] = f_acondicionar(signal)
    fs = 100; 
    [b, a] = butter(4, [2, 49] / (fs / 2), 'bandpass');
    filtered_signal = filtfilt(b, a, signal);
    rectified_signal = abs(filtered_signal);
    %smoothed_signal = movmean(rectified_signal, 50);
end