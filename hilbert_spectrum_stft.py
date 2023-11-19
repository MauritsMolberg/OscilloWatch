"""
STFT variant of HHT made by ChatGPT. Will probably not be used for anything.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.signal import hilbert
from scipy.interpolate import RegularGridInterpolator

def compute_hilbert_spectrum(signal, fs, freq_resolution=100, nperseg=256):
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg)
    analytic_signal = hilbert(Zxx.real, axis=1)
    amplitude_signal = np.abs(analytic_signal)

    # Resample the amplitude signal to match freq_resolution
    f_orig, t_orig = f, t
    f_resampled = np.linspace(f_orig[0], f_orig[-1], freq_resolution)
    t_resampled = np.linspace(t_orig[0], t_orig[-1], amplitude_signal.shape[1])

    t_mesh, f_mesh = np.meshgrid(t_resampled, f_resampled)
    interpolator = RegularGridInterpolator((f_orig, t_orig), amplitude_signal, method='linear')
    amplitude_signal_resampled = interpolator(np.column_stack((f_mesh.ravel(), t_mesh.ravel()))).reshape(t_mesh.shape)

    return t_mesh, f_mesh, amplitude_signal_resampled

def hilbert_spectrum_stft(signals, fs, freq_resolution=100, nperseg=256):
    total_amplitude_signal = None

    for signal in signals:
        t_mesh, f_mesh, amplitude_signal = compute_hilbert_spectrum(signal, fs, freq_resolution, nperseg)
        if total_amplitude_signal is None:
            total_amplitude_signal = amplitude_signal
        else:
            total_amplitude_signal += amplitude_signal

    fig, ax = plt.subplots()
    c = ax.pcolormesh(t_mesh, f_mesh, total_amplitude_signal, shading='auto')
    fig.colorbar(c, ax=ax)
    plt.show()

if __name__ == "__main__":
    start = 0
    end = 5
    fs = 10000
    t = np.arange(start, end, 1/fs)  # Time vector


    signal1 = 10*np.exp(.2*t)*np.cos(2.4*np.pi*t) + 8*np.exp(-.1*t)*np.cos(np.pi*t)

    # To plot the combined Hilbert spectrum for a list of signals
    hilbert_spectrum_stft([signal1], fs, nperseg=100)
    def f(t):
        #return 10*np.exp(.2*t)*np.cos(2.4*np.pi*t) + 8*np.exp(-.1*t)*np.cos(np.pi*t)
        #return 3*np.sin(5*np.pi*t)
        return np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)

