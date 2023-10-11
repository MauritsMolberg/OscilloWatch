import numpy as np
import matplotlib.pyplot as plt
from emd import emd
from scipy.signal import hilbert

def hht_single(signal, freq_resolution = 1000, max_omega = "auto"):
    hilbert_signal = hilbert(signal)
    omega_signal = np.gradient(np.angle(hilbert_signal))/2/np.pi
    amplitude_signal = np.abs(hilbert_signal)

    if max_omega == "auto":
        max_omega = np.amax(omega_signal)

    xAxis, omegaAxis = np.arange(0, len(signal), 1), np.linspace(0, max_omega, freq_resolution)

    hilbert_spectrum = np.zeros((len(omegaAxis), len(xAxis)))

    for i in range(len(xAxis)):
        for j in range(len(omegaAxis)):
            if abs(omegaAxis[j] - omega_signal[i]) < .0005:
                hilbert_spectrum[j][i] += amplitude_signal[i]

    xAxis_mesh, omegaAxis_mesh = np.meshgrid(xAxis, omegaAxis)
    fig, ax = plt.subplots()
    c = ax.pcolormesh(xAxis_mesh, omegaAxis_mesh, hilbert_spectrum, shading="auto")
    fig.colorbar(c, ax=ax)
    plt.show()


    return hilbert_spectrum



def hht(signal_list, freq_resolution = 1000):
    hilbert_signal_list =  []
    omega_signal_list = []
    amplitude_signal_list = []


    if not isinstance(signal_list[0], np.ndarray):
        signal_list = [signal_list]

    for signal in signal_list:
        hilbert_signal = hilbert(signal)
        hilbert_signal_list.append(hilbert_signal)
        omega_signal_list.append(np.gradient(np.angle(hilbert_signal))/2/np.pi)
        amplitude_signal_list.append(np.abs(hilbert_signal))

    max_omega = np.amax(omega_signal_list)
    print(max_omega)

    xAxis, omegaAxis = np.arange(0, len(signal_list[0]), 1), np.linspace(0, max_omega, freq_resolution)

    hilbert_spectrum = np.zeros((len(omegaAxis), len(xAxis)))

    for k in range(len(signal_list)):
        for i in range(len(xAxis)):
            for j in range(len(omegaAxis)):
                if abs(omegaAxis[j] - omega_signal_list[k][i]) < .00005:
                    hilbert_spectrum[j][i] += amplitude_signal_list[k][i]

    xAxis_mesh, omegaAxis_mesh = np.meshgrid(xAxis, omegaAxis)
    fig, ax = plt.subplots()
    c = ax.pcolormesh(xAxis_mesh, omegaAxis_mesh, hilbert_spectrum, shading="auto")
    fig.colorbar(c, ax=ax)
    plt.show()

    return hilbert_spectrum




def f(t):
    return 10*np.exp(.2*t)*np.cos(2.4*np.pi*t) + 8*np.exp(-.1*t)*np.cos(np.pi*t)

start = 0
end = 5
fs = 100
input_signal1 = f(np.arange(start, end, 1/fs))

#Random (reproducable) signal
np.random.seed(0)
input_signal2 = np.random.randn(500)


res1, imf_list1 = emd(input_signal1, max_imfs=4, mirror_padding_fraction=1)
res2, imf_list2 = emd(input_signal2)


hht(imf_list1)