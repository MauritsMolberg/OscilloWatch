import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, stft, find_peaks
from time import time
from OscilloWatch.AnalysisSettings import AnalysisSettings
from OscilloWatch.EMD import EMD


class HHT:
    """
    Class for performing the Hilbert-Huang Transform on a signal segment.
    """

    def __init__(self, input_signal, settings: AnalysisSettings):
        """
        Constructor for HHT class. Initializes variables, but does not perform the actual HHT algorithm.

        :param input_signal: Signal to be decomposed.
        :type input_signal: numpy.ndarray or list
        :param settings: Object containing the settings for the HHT algorithm, and the other methods that will be used
         in the signal analysis.
        :type settings: AnalysisSettings
        """
        self.settings = settings
        self.input_signal = input_signal

        self.emd = EMD(self.input_signal, self.settings)

        self.samples_to_remove_start = 0
        self.samples_to_remove_end = 0

        self.freq_signal_list = []
        self.amplitude_signal_list = []
        self.freq_axis = np.array([])
        self.hilbert_spectrum = np.array([])

        self.runtime = 0

    def calc_hilbert_spectrum(self, signal_list):
        """
        Calculates the Hilbert spectrum on a signal or list of signals. Updates the hilbert_spectrum and freq_axis
        variables of the HHT object.

        :param signal_list: Either a list of signals or a single signal, which the Hilbert spectrum is to be calculated
            for.
        :type signal_list: list[list] or list[numpy.ndarray] or list[float] or numpy.ndarray[float]
        :return: None
        """
        # Handling the case of signal_list being only one signal (puts it in a list).
        # Not relevant if performed on a list of IMFs.
        if not isinstance(signal_list[0], np.ndarray) and not isinstance(signal_list[0], list):
            signal_list = [signal_list]

        imf_count = 1  # For test plotting/printing
        for signal in signal_list:
            #print("IMF", imf_count)

            if self.settings.hht_split_signal_freq_change_toggle:
                # Split the signal where there is an abrupt change in frequency
                split_signal = split_signal_freq_change(signal,
                                                        nperseg=self.settings.hht_split_signal_freq_change_nperseg,
                                                        fs=self.settings.fs,
                                                        threshold=self.settings.hht_split_signal_freq_change_threshold)
                # Calculate Hilbert transform on each portion of the split signal to find inst. freq. and amplitude.
                analytic_signal = np.array([])
                freq_signal = np.array([])
                amplitude_signal = np.array([])
                for portion in split_signal:
                    analytic_portion = hilbert(portion)
                    freq_portion = np.gradient(np.unwrap(np.angle(analytic_portion))) * self.settings.fs / (2*np.pi)
                    freq_portion = moving_average(freq_portion,
                                                  window_size=self.settings.hht_frequency_moving_avg_window)
                    amplitude_portion = np.abs(analytic_portion)
                    # Join calculated freq. and amp. for the portions together again
                    analytic_signal = np.concatenate((analytic_signal, analytic_portion))
                    freq_signal = np.concatenate((freq_signal, freq_portion))
                    amplitude_signal = np.concatenate((amplitude_signal, amplitude_portion))
            else:
                analytic_signal = hilbert(signal)
                freq_signal = np.gradient(np.unwrap(np.angle(analytic_signal))) * self.settings.fs / (2*np.pi)
                freq_signal = moving_average(freq_signal,
                                             window_size=self.settings.hht_frequency_moving_avg_window)
                amplitude_signal = np.abs(analytic_signal)

            #plt.figure()
            #plt.plot(np.unwrap(np.angle(analytic_signal)))
            #plt.plot(freq_signal)
            #plt.plot()

            # Remove padding, if specified.
            if self.samples_to_remove_start > 0:
                freq_signal = freq_signal[self.samples_to_remove_start:]
                amplitude_signal = amplitude_signal[self.samples_to_remove_start:]
            if self.samples_to_remove_end > 0:
                freq_signal = freq_signal[:-self.samples_to_remove_end]
                amplitude_signal = amplitude_signal[:-self.samples_to_remove_end]

            #fig, axes = plt.subplots(2, 1, figsize=(8, 8))
            #axes[0].plot(amplitude_signal)
            #axes[0].set_title(f"Amplitude IMF {imf_count} (unfiltered)")
            #axes[1].plot(freq_signal, color="red")
            #axes[1].set_title(f"Frequency IMF {imf_count} (unfiltered)")

            imf_count += 1

            amplitude_signal = moving_average(amplitude_signal,
                                              window_size=self.settings.hht_amplitude_moving_avg_window)

            # Store inst. freq. and amplitude to lists
            self.freq_signal_list.append(freq_signal)
            self.amplitude_signal_list.append(amplitude_signal)

        # Set frequency axis max/min
        if self.settings.maximum_frequency is None:
            freq_axis_max = np.amax(self.freq_signal_list)
        else:
            freq_axis_max = self.settings.maximum_frequency
        if self.settings.minimum_frequency is None:
            freq_axis_min = 1/(self.settings.segment_length_time
                               + self.settings.extension_padding_time_start
                               + self.settings.extension_padding_time_end)
        else:
            freq_axis_min = self.settings.minimum_frequency

        # Cannot construct meaningful Hilbert spectrum if the lowest detectable frequency is higher than the highest
        # measured frequency
        if freq_axis_min >= freq_axis_max:
            self.freq_axis = np.zeros(1)
            self.hilbert_spectrum = np.zeros([1, 1])
            #print("No frequencies above minimum limit detected.")
            return

        if freq_axis_max < self.settings.hht_frequency_resolution:  # Give frequency axis length of at least 1
            freq_axis_max = self.settings.hht_frequency_resolution

        # Construct frequency axis, making sure it has one "extra" element at the end, to ensure that there is an extra
        # row at the top of the Hilbert spectrum (for the spectrum analysis algorithm to work properly)
        self.freq_axis = np.arange(freq_axis_min, freq_axis_max + self.settings.hht_frequency_resolution,
                                   self.settings.hht_frequency_resolution)

        # Constructing spectrum:
        self.hilbert_spectrum = np.zeros((len(self.freq_axis), len(self.amplitude_signal_list[0])))
        for k in range(len(signal_list)): # For each signal (IMF)
            for i in range(len(self.amplitude_signal_list[k])): # For each sample in segment
                for j in range(len(self.freq_axis)): # For each frequency
                    if (abs(self.freq_axis[j] - self.freq_signal_list[k][i]) < self.settings.hht_frequency_resolution/2
                       and self.amplitude_signal_list[k][i] > self.settings.minimum_amplitude):
                        # Using maximum of amplitude values in case of overlap, instead of adding together
                        self.hilbert_spectrum[j][i] = max(self.amplitude_signal_list[k][i], self.hilbert_spectrum[j][i])

        # Make spectrum a 1x1 array if it is empty (consists of only zeros), to save memory and storage space
        if np.amax(self.hilbert_spectrum) == 0.0:
            self.freq_axis = np.zeros(1)
            self.hilbert_spectrum = np.zeros([1, 1])
            #print("Empty Hilbert spectrum")
            return

        # Delete rows with only zeros from the top of the spectrum, to reduce its size
        row_remove_count = -1
        for freq in reversed(self.hilbert_spectrum):
            if np.amax(freq) > 0.0:
                break
            else:
                row_remove_count += 1
        if 1 < row_remove_count < len(self.freq_axis):
            self.freq_axis = self.freq_axis[:-row_remove_count]
            self.hilbert_spectrum = self.hilbert_spectrum[:-row_remove_count]

        return

    def full_hht(self):
        """
        Performs the HHT algorithm on the HHT object's input signal. Performs EMD to find IMFs, then calculates the
        Hilbert spectrum on the list of IMFs using calc_hilbert_spectrum().

        :return: None
        """
        start_time = time()

        self.emd.perform_emd()  # Calculate imf_list and residual

        # Use residual if no IMFs were found
        if not self.emd.imf_list:
            self.emd.imf_list = [self.emd.residual]

        # Calculate how much to remove after Hilbert transform in Hilbert Spectrum calculation:
        if self.settings.remove_padding_after_hht and not self.settings.remove_padding_after_emd:
            mirror_fraction_removal = int(self.settings.mirror_padding_fraction*len(self.input_signal))
            self.samples_to_remove_start = mirror_fraction_removal + self.settings.extension_padding_samples_start
            self.samples_to_remove_end = mirror_fraction_removal + self.settings.extension_padding_samples_end
        # Else: Both kept at 0

        self.calc_hilbert_spectrum(self.emd.imf_list)  # Calculate hilbert_spectrum and freq_axis

        self.runtime = time() - start_time
        if self.settings.print_hht_time:
            print(f"HHT completed in {self.runtime:.3f} seconds.")
        return

    def plot_hilbert_spectrum(self, show = False):
        """
        Plots the HHT object's Hilbert spectrum as a heatmap.

        :param bool show: Specifies whether the plt.show() should be run at the end of the function.
        :return: None
        """
        spec_copy = np.copy(self.hilbert_spectrum)
        # Setting zeros to None (for clearer plot)
        for i in range(len(spec_copy)):
            for j in range(len(spec_copy[0])):
                if not spec_copy[i][j]:
                    spec_copy[i][j] = None
        xAxis = np.linspace(0, len(spec_copy[0])/self.settings.fs, len(spec_copy[0]))
        xAxis_mesh, freq_axis_mesh = np.meshgrid(xAxis, self.freq_axis)
        fig, ax = plt.subplots(figsize=(12, 7))
        #ax.set_title("Hilbert spectrum")
        ax.set_xlabel("Time [s]", fontsize=16)
        ax.set_ylabel("Frequency [Hz]", fontsize=16)
        c = ax.pcolormesh(xAxis_mesh, freq_axis_mesh, spec_copy, shading="auto")
        cbar = fig.colorbar(c, ax=ax, fraction=.05)
        cbar.set_label("Amplitude", fontsize=16, labelpad=8)
        plt.tight_layout()
        if show:
            plt.show()


def moving_average(signal, window_size=5):
    """
    Smooths out the signal using a moving average filter.

    :param signal: Signal to be smoothed out.
    :type signal: numpy.ndarray or list
    :param window_size: Size of the moving window.
    :type window_size: int
    :return: Smoothed signal.
    :rtype: numpy.ndarray
    :raises ValueError: If the window size is an even number.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd number.")

    half_window = window_size // 2
    smoothed_signal = np.zeros_like(signal, dtype=float)

    for i in range(len(signal)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(signal), i + half_window + 1)
        smoothed_signal[i] = np.mean(signal[start_idx:end_idx])

    return smoothed_signal


def split_signal_freq_change(signal, threshold=0.5, fs=50, nperseg=256):
    """
    Performs STFT on the signal, finds the dominant frequency and its derivative to estimate where the dominant
    frequency changes abruptly. Splits the signal in these places and returns a list containing each of the portions of
    the original signal. Intended to be used on IMFs, to ensure a well-behaving Hilbert transform on most of the IMF,
    even if it abruptly changes frequency.

    :param signal: Signal to be split at abrupt frequency changes.
    :type signal: numpy.ndarray or list
    :param threshold: The value the peak in derivative of dominant must exceed for the function to split the signal.
    :type threshold: float
    :param fs: Sampling frequency. Number of samples per second. Used to give accurate frequencies.
    :type fs: int
    :param nperseg: Number of samples in each window of the STFT. Higher means better frequency resolution (and thereby
     more accurate dominant frequency, making it easier to set a universial threshold), at the cost of worse time
     resolution, making the location of the abrupt frequency changes less accurate.
    :type nperseg: int

    :return: List of signal portions, split where the dominant frequency changes abruptly.
    :rtype: list
    """
    split_signal = []

    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg)
    dominant_frequencies = np.argmax(np.abs(Zxx), axis=0)
    freq_gradient = np.gradient(dominant_frequencies)
    #plt.figure()
    #plt.plot(freq_gradient)
    peaks, _ = find_peaks(np.abs(freq_gradient))

    remaining = np.copy(signal)
    ind_correction = 0  # Used to get correct indexing for the "remaining" array when it has been shrunk
    for ind in peaks:
        if abs(freq_gradient[ind]) > threshold:
            #print("Split:", ind * nperseg//2/fs)
            split_signal.append(remaining[:(ind-ind_correction) * nperseg//2])
            remaining = remaining[(ind-ind_correction) * nperseg//2:]
            ind_correction = ind
    split_signal.append(remaining)

    return split_signal
