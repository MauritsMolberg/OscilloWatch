import numpy as np
import matplotlib.pyplot as plt
from emd import *
from scipy.signal import hilbert
from time import time

def calc_hilbert_spectrum_single(signal, freq_resolution = 1000, max_omega = "auto"):
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



def calc_hilbert_spectrum(signal_list, freq_resolution = 1e-4, freq_tol = "res"):
    hilbert_signal_list =  []
    omega_signal_list = []
    amplitude_signal_list = []

    if freq_tol == "res":
        freq_tol = freq_resolution

    if freq_tol == "2res":
        freq_tol = 2*freq_resolution

    # Handling the case of signal_list being only one signal (puts it in a list)
    if not isinstance(signal_list[0], np.ndarray) and not isinstance(signal_list[0], list):
        signal_list = [signal_list]

    for signal in signal_list:
        hilbert_signal = hilbert(signal)
        hilbert_signal_list.append(hilbert_signal)
        omega_signal_list.append(np.gradient(np.angle(hilbert_signal))/2/np.pi)
        amplitude_signal_list.append(np.abs(hilbert_signal))

    max_omega = np.amax(omega_signal_list)

    plt.figure()
    plt.plot(omega_signal_list[0])

    xAxis = np.arange(0, len(signal_list[0]), 1)
    omegaAxis = np.linspace(0, max_omega, int(max_omega/freq_resolution))

    hilbert_spectrum = np.zeros((len(omegaAxis), len(xAxis)))

    for k in range(len(signal_list)):
        for i in range(len(xAxis)):
            for j in range(len(omegaAxis)):
                if abs(omegaAxis[j] - omega_signal_list[k][i]) < freq_tol:
                    hilbert_spectrum[j][i] += amplitude_signal_list[k][i]

    return hilbert_spectrum, omegaAxis


def hht(signal,
        freq_resolution = 1e-4,
        freq_tol = "res",
        sd_tolerance=.2,
        max_imfs=10,
        max_sifting_iterations = 30,
        mirror_padding_fraction = .5,
        print_sifting_details = False):
    imf_list, res = emd(signal, sd_tolerance, max_imfs, max_sifting_iterations, mirror_padding_fraction, print_sifting_details)
    return calc_hilbert_spectrum(imf_list, freq_resolution, freq_tol)


def plot_hilbert_spectrum(hilbert_spectrum, omegaAxis, show = True):
    xAxis = np.arange(0, len(hilbert_spectrum[0]), 1)
    xAxis_mesh, omegaAxis_mesh = np.meshgrid(xAxis, omegaAxis)
    fig, ax = plt.subplots()
    c = ax.pcolormesh(xAxis_mesh, omegaAxis_mesh, hilbert_spectrum, shading="auto")
    fig.colorbar(c, ax=ax)
    if show:
        plt.show()

if __name__ == "__main__":
    def f(t):
        #return 10*np.exp(.2*t)*np.cos(2.4*np.pi*t) + 8*np.exp(-.1*t)*np.cos(np.pi*t)
        return 3*np.sin(5*np.pi*t)

    start = 0
    end = 5
    fs = 100
    input_signal1 = f(np.arange(start, end, 1/fs))

    #Random (reproducable) signal
    np.random.seed(0)
    input_signal2 = np.random.randn(500)


    imf_list, res = emd(input_signal1)
    #imf_list, res = emd(input_signal2)

    #plot_emd_results(input_signal1, imf_list, res)

    start_time = time()
    hilbert_spectrum, omegaAxis = hht(input_signal1, freq_resolution=1e-4, freq_tol="2res")
    print("HHT time:", time()-start_time)
    #hilbert_spectrum, omegaAxis = calc_hilbert_spectrum(input_signal1)
    plot_hilbert_spectrum(hilbert_spectrum, omegaAxis)

#Implement "signal extension" for IMFs

