import numpy as np
import matplotlib.pyplot as plt
def plot_hilbert_spectrum(hilbert_spectrum, freqAxis, sampling_freq, show = True):
    xAxis = np.linspace(0, len(hilbert_spectrum[0])/sampling_freq, len(hilbert_spectrum[0]))
    xAxis_mesh, freqAxis_mesh = np.meshgrid(xAxis, freqAxis)
    fig, ax = plt.subplots(figsize=(12, 7))
    #ax.set_title("Hilbert spectrum")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    c = ax.pcolormesh(xAxis_mesh, freqAxis_mesh, hilbert_spectrum, shading="auto")
    fig.colorbar(c, ax=ax, fraction=.05)
    plt.tight_layout()
    if show:
        plt.show()

start = 0
stop = 10
fs = 200
tAxis = np.arange(start, stop, 1/fs)
def a(t):
    return 5*np.exp(.1*t)

def f(t):
    return a(t)*np.cos(3*2*np.pi*t)

fig, ax = plt.subplots(figsize=(10.7, 7))
ax.plot(tAxis, f(tAxis))
ax.set_xlabel("Time [s]")
plt.tight_layout()

freqAxis = np.linspace(0, 3.2, 80)
hilbert_spectrum = np.empty((len(freqAxis), len(tAxis)))
hilbert_spectrum[:] = np.nan
hilbert_spectrum[74] = a(tAxis)

plot_hilbert_spectrum(hilbert_spectrum, freqAxis, fs)