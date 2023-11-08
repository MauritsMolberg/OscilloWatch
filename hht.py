import numpy as np
import matplotlib.pyplot as plt
from emd import emd, plot_emd_results
from scipy.signal import hilbert
from time import time

def calc_hilbert_spectrum(signal_list,
                          amp_tol = 0.0,
                          freq_resolution = 1e-4,
                          freq_tol = "f_res",
                          samples_to_remove_start = 0,
                          samples_to_remove_end = 0):
    hilbert_signal_list =  []
    freq_signal_list = []
    amplitude_signal_list = []

    if freq_tol == "f_res":
        freq_tol = freq_resolution

    if freq_tol == "2f_res":
        freq_tol = 2*freq_resolution

    # Handling the case of signal_list being only one signal (puts it in a list)
    if not isinstance(signal_list[0], np.ndarray) and not isinstance(signal_list[0], list):
        signal_list = [signal_list]

    for signal in signal_list:
        hilbert_signal = hilbert(signal)
        freq_signal = np.gradient(np.angle(hilbert_signal))/2/np.pi

        if samples_to_remove_start > 0:
            hilbert_signal = hilbert_signal[samples_to_remove_start:]
            freq_signal = freq_signal[samples_to_remove_start:]
        if samples_to_remove_end > 0:
            hilbert_signal = hilbert_signal[:-samples_to_remove_end]
            freq_signal = freq_signal[:-samples_to_remove_end]

        hilbert_signal_list.append(hilbert_signal)
        freq_signal_list.append(freq_signal)
        amplitude_signal_list.append(np.abs(hilbert_signal))

    max_freq = np.amax(freq_signal_list)

    tAxis = np.arange(0, len(hilbert_signal_list[0]), 1)
    freqAxis = np.linspace(0, max_freq, int(max_freq/freq_resolution))

    hilbert_spectrum = np.zeros((len(freqAxis), len(tAxis)))

    for k in range(len(signal_list)):
        for i in range(len(tAxis)):
            for j in range(len(freqAxis)):
                if abs(freqAxis[j] - freq_signal_list[k][i]) < freq_tol and amplitude_signal_list[k][i] > amp_tol:
                    hilbert_spectrum[j][i] += amplitude_signal_list[k][i]

    # Removing frequencies with zero amplitude across time axis from hilbert_spectrum and freqAxis, until an amplitude
    # > 0 is found.
    remove_count = 0
    for freq in reversed(hilbert_spectrum):
        if np.amax(freq) > 0.0:
            break
        else:
            remove_count += 1

    if remove_count > 0:
        print("remove count:", remove_count)
        freqAxis = freqAxis[:-remove_count]
        hilbert_spectrum = hilbert_spectrum[:-remove_count]

    return hilbert_spectrum, freqAxis


def hht(signal,
        amp_tol = 0.0,
        freq_resolution = 1e-4,
        freq_tol = "f_res",
        sd_tolerance=.2,
        max_imfs=10,
        max_sifting_iterations = 30,
        mirror_padding_fraction = .5,
        remove_padding_after_emd = False,
        remove_padding_after_hht = True,
        print_emd_sifting_details = False,
        print_emd_time = False,
        print_hht_time = False):

    start_time = time()

    imf_list, res = emd(signal,
                        sd_tolerance=sd_tolerance,
                        max_imfs=max_imfs,
                        max_sifting_iterations=max_sifting_iterations,
                        mirror_padding_fraction=mirror_padding_fraction,
                        remove_padding=remove_padding_after_emd,
                        print_sifting_details=print_emd_sifting_details,
                        print_time=print_emd_time)

    # Calculate how much to remove after Hilbert transform in Hilbert Spectrum calculation:
    if remove_padding_after_hht and not remove_padding_after_emd: # Will not remove after hht if already removed after emd
        samples_to_remove_start = int(mirror_padding_fraction*len(signal))
        samples_to_remove_end = samples_to_remove_start
    else:
        samples_to_remove_start = 0
        samples_to_remove_end = 0

    hilbert_spectrum, freqAxis = calc_hilbert_spectrum(imf_list,
                                amp_tol=amp_tol,
                                freq_resolution = freq_resolution,
                                freq_tol = freq_tol,
                                samples_to_remove_start = samples_to_remove_start,
                                samples_to_remove_end = samples_to_remove_end)

    if print_hht_time:
        print("HHT completed in", round(time() - start_time, 3), "seconds.")

    return hilbert_spectrum, freqAxis


def plot_hilbert_spectrum(hilbert_spectrum, freqAxis, show = True):
    xAxis = np.arange(0, len(hilbert_spectrum[0]), 1)
    xAxis_mesh, freqAxis_mesh = np.meshgrid(xAxis, freqAxis)
    fig, ax = plt.subplots()
    c = ax.pcolormesh(xAxis_mesh, freqAxis_mesh, hilbert_spectrum, shading="auto")
    fig.colorbar(c, ax=ax)
    if show:
        plt.show()

if __name__ == "__main__":
    def f(t):
        return 10*np.exp(.2*t)*np.cos(2.4*np.pi*t) + 8*np.exp(-.1*t)*np.cos(np.pi*t)
        #return 3*np.sin(5*np.pi*t)

    start = 0
    end = 5
    fs = 50
    input_signal1 = f(np.arange(start, end, 1/fs))

    #Random (reproducable) signal
    np.random.seed(0)
    input_signal2 = np.random.randn(500)


    imf_list, res = emd(input_signal1, remove_padding=True)

    plot_emd_results(input_signal1, imf_list, res, show=False)

    hilbert_spectrum, freqAxis = hht(input_signal1,
                                     freq_resolution=1e-5,
                                     freq_tol="2f_res",
                                     print_hht_time=True,
                                     print_emd_time=True)


    plot_hilbert_spectrum(hilbert_spectrum, freqAxis, show=False)


    plt.show()


