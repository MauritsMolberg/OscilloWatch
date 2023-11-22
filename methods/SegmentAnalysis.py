import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from methods.HHT import HHT
from methods.EMD import EMD
from methods.AnalysisSettings import AnalysisSettings


class SegmentAnalysis:
    """
    Class for determining the different oscillating modes and their damping in a signal segment by performing HHT and
    interpreting the Hilbert spectrum.
    """

    def __init__(self, input_signal, settings: AnalysisSettings):
        """
        Constructor for the SegmentAnalysis class. Initializes variables, but does not perform the actual analysis.

        :param input_signal: Signal segment to be analyzed.
        :type input_signal: numpy.ndarray or list
        :param settings: Object containing the settings for the different algorithms used in the signal analysis.
        :type settings: AnalysisSettings
        """
        self.input_signal = input_signal
        self.settings = settings

        self.hht = HHT(self.input_signal, self.settings)


        self.damping_info_list = []

    def interpolate_signal(self, signal, damping_info):
        """
        Fills in zero-values of a signal using cubic spline interpolation. Deletes sequences of non-zero values that are
        too short to be assumed to be meaningful. Disregards all zero-elements before and after the first and last non-
        zero element. Updates info about where this sequence starts and ends in the dict damping_info, which is a
        parameter. Intended to be used on a row in a Hilbert spectrum.

        :param signal: Amplitude curve containing zeros. Intended to be a single or combined row from a
            Hilbert spectrum containing the amplitude of a single oscillating mode.
        :type signal: numpy.ndarray
        :param damping_info: Dict containing info about the oscillating mode that is to be analyzed.
        :type damping_info: dict

        :return: Copy of the portion of the input signal that starts at the first non-zero and ends at the last, with
            zero-values in-between non-zero values filled in.
        :rtype: numpy.ndarray
        """
        # Find the indices of non-zero values
        non_zero_indices = np.nonzero(signal)[0]
        non_zero_indices = remove_short_consecutive_sequences(non_zero_indices,
                                                              self.settings.minimum_consecutive_non_zero_length)
        # If there are no non-zero values, return the original signal
        if not len(non_zero_indices):
            return signal

        # Find the first and last index that is not zero
        first_non_zero_index = non_zero_indices[0]
        damping_info["start_time"] = first_non_zero_index/self.settings.fs
        last_non_zero_index = non_zero_indices[-1] + 1
        damping_info["end_time"] = last_non_zero_index/self.settings.fs

        # Ignore all zero values before the first and after the last non-zero value
        cropped_signal = signal[first_non_zero_index:last_non_zero_index]

        # Recalculate non_zero_indices based on the cropped signal
        non_zero_indices = np.nonzero(cropped_signal)[0]
        damping_info["interpolated_fraction"] = 1 - len(non_zero_indices)/len(cropped_signal)

        # Get the corresponding non-zero values and their indices
        non_zero_values = [cropped_signal[i] for i in non_zero_indices]

        # Create a cubic spline interpolation function
        spline = CubicSpline(non_zero_indices, non_zero_values)

        # Generate the new signal with interpolated values
        new_signal = []
        for i, value in enumerate(cropped_signal):
            if not value:
                new_signal.append(float(spline(i)))
            else:
                new_signal.append(value)
        return new_signal

    def analyze_frequency_band(self, amp_curve, n, k):
        """
        Analyzes the damping of a frequency band by curve fitting an amplitude curve to a decaying exponential curve.
        Removes zero-values before and after the first and last non-zero value and interpolates the remaining curve
        before curve fitting. Stores the following information about the frequency band in a dict that is added to the
        DampingAnalysis object's damping_info_list variable:

        Frequency range: Estimated frequency range of the frequency band

        Start and end time: When the oscillations in the frequency band start and stop, according to the spectrum

        Non-zero fraction: Fraction of the whole amplitude curve that is not zero

        Initial and final amplitude: Measured amplitude of the first and last non-zero value

        Initial amplitude estimate: Initial amplitude of the fitted exponential curve

        Decay rate: Decay rate of the fitted exponential curve

        Damping ratio: Estimated damping ratio based on the decay rate and central frequency of the frequency band

        Interpolated fraction: Fraction of the amplitude curve that has been interpolated.

        Coefficient of variation: Standard deviation between the interpolated amplitude curve and the fitted curve,
        divided by the mean of the interpolated curve

        Note: String that stores some info about the process, such as if the damping info is skipped

        :param numpy.ndarray amp_curve: Single or combined row from Hilbert spectrum that is being analyzed.
        :param int n: Index of the bottom-most row in the frequency band that is being analyzed
        :param int k: Number of rows that are included in this frequency band in addition to the bottom one.
        :return: None
        """
        non_zero_fraction = np.count_nonzero(amp_curve)/len(amp_curve)
        damping_info = {
            "frequency_range": (self.hht.freq_axis[n], self.hht.freq_axis[n+k]),
            "start_time": 0.0,  # Start and end time are set in the interpolate_signal function
            "end_time": 0.0,
            "non_zero_fraction": non_zero_fraction,
            "initial_amplitude": 0.0,
            "final_amplitude": 0.0,
            "initial_amplitude_estimate": 0.0,
            "decay_rate": 0.0,
            "damping_ratio": 0.0,
            "interpolated_fraction": 0.0,
            "coefficient_of_variation": 0.0,
            "note": ""
        }
        interp_amp_curve = self.interpolate_signal(amp_curve, damping_info)

        # Curve fit to find approximate initial amplitude and decay rate
        time_points = np.linspace(0, len(interp_amp_curve)/self.settings.fs, len(interp_amp_curve))
        popt, _ = 0, 0
        try:
            popt, _ = curve_fit(exponential_decay_model, time_points, interp_amp_curve, maxfev=1500)
        except RuntimeError:
            damping_info["note"] = "Skipped. Could not fit curve."
            self.damping_info_list.append(damping_info)
            return
        A, decay_rate = popt[0], popt[1]

        damping_info["initial_amplitude"] = interp_amp_curve[0]
        damping_info["final_amplitude"] = interp_amp_curve[-1]
        damping_info["initial_amplitude_estimate"] = A
        row_ind = n + int((k+1)//2)
        damping_info["decay_rate"] = decay_rate
        damping_info["damping_ratio"] = decay_rate/(np.sqrt(decay_rate**2 + (2*np.pi*self.hht.freq_axis[row_ind])**2))
        damping_info["coefficient_of_variation"]\
            = np.std(interp_amp_curve - exponential_decay_model(time_points, A, decay_rate)) / np.mean(interp_amp_curve)
        print(damping_info, "\n")
        fig, ax = plt.subplots()
        ax.plot(interp_amp_curve)
        ax.plot(exponential_decay_model(time_points, A, decay_rate))
        self.damping_info_list.append(damping_info)
        return


    def damping_analysis(self):
        """
        Analyzes the damping of different components of the signal segment by performing HHT and using the developed algorithm
        to analyze the Hilbert spectrum.
        :return: None
        """
        self.hht.full_hht()  # Calculate hilbert_spectrum and freq_axis

        n = 0
        while n < len(self.hht.freq_axis):
            # k = 0
            non_zero_count = np.count_nonzero(self.hht.hilbert_spectrum[n])
            if not non_zero_count:
                n += 1
                continue
            combined_row = self.hht.hilbert_spectrum[n]

            k = 1
            while n + k < len(self.hht.freq_axis):
                non_zero_count_old = non_zero_count
                combined_row = np.maximum(combined_row, self.hht.hilbert_spectrum[n+k])
                non_zero_count = np.count_nonzero(combined_row)
                if non_zero_count - non_zero_count_old > self.settings.minimum_non_zero_improvement \
                    or np.count_nonzero(self.hht.hilbert_spectrum[n+k]) \
                        > self.settings.minimum_total_non_zero_fraction*len(self.input_signal):
                    k += 1
                else:
                    non_zero_count = non_zero_count_old
                    break
            if non_zero_count/len(self.input_signal) > self.settings.minimum_total_non_zero_fraction:
                self.analyze_frequency_band(combined_row, n, k-1)
            n += k

def remove_short_consecutive_sequences(non_zero_indices, min_consecutive_length):
    """
    Removes consecutive sequences of integers from non_zero_indices parameter if they are shorter than the specified
    minimum consecutive length.

    :param numpy.ndarray non_zero_indices: Contains indices of a signal's non-zero values.
    :param int min_consecutive_length: The length a consecutive sequence of indices must exceed to not be removed from
        the returned version of non_zero_indices.

    :return: Updated version of non_zero_indices with the short consecutive sequences removed.
    :rtype: numpy.ndarray
    """
    if len(non_zero_indices) < 2:
        # No consecutive sequences to remove if there are fewer than 2 elements
        return non_zero_indices

    # Initialize variables
    current_sequence = [non_zero_indices[0]]
    prev_index = non_zero_indices[0]
    updated_indices = []

    # Iterate through the sorted indices
    for index in non_zero_indices[1:]:
        # Check if the current index is consecutive to the previous index
        if index == prev_index + 1:
            current_sequence.append(index)
        else:
            # Remove indices from the current sequence if it's shorter than the minimum length
            if len(current_sequence) >= min_consecutive_length:
                updated_indices.extend(current_sequence)
            # Start a new sequence
            current_sequence = [index]

        # Update the previous index
        prev_index = index

    # Check the last sequence
    if len(current_sequence) >= min_consecutive_length:
        updated_indices.extend(current_sequence)

    return np.array(updated_indices)

def exponential_decay_model(t, A, k):
    """
    Model for exponential decay curve used for curve fitting amplitude curve to.

    :param numpy.ndarray t: Time-points
    :param float A: Initial amplitude of curve
    :param float k:
    :return:
    """
    return A * np.exp(-k * t)
