from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

start = 0
end = 5
fs = 50
t = np.arange(start, end, 1/fs)


def f(t):
    return np.sin(t)


signal = f(t)

upper_peaks, _ = find_peaks(signal)
lower_peaks, _ = find_peaks(-signal)

upper_peaks = np.insert(upper_peaks, 0, 0)
upper_peaks = np.append(upper_peaks, len(signal) - 1)
lower_peaks = np.insert(lower_peaks, 0, 0)
lower_peaks = np.append(lower_peaks, len(signal) - 1)


plt.figure()
plt.plot(t, signal)
plt.plot(upper_peaks/fs, signal[upper_peaks],'o',label = 'Upper peaks')
plt.plot(lower_peaks/fs, signal[lower_peaks],'o',label = 'Lower peaks')

plt.show()