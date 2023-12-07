import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from time import time
from methods.HHT import HHT
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


        self.oscillation_info_list = []

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
        
        start_index = non_zero_indices[0]
        end_index = non_zero_indices[-1] + 1
        
        first_max_index = np.argmax(signal)
        if self.settings.start_amp_curve_at_peak and first_max_index <= len(signal[start_index:end_index])/2:
            start_index = first_max_index

        damping_info["Start time"] = start_index/self.settings.fs
        damping_info["End time"] = end_index/self.settings.fs

        # Ignore all zero values before the first and after the last non-zero value
        cropped_signal = signal[start_index:end_index]

        # Recalculate non_zero_indices based on the cropped signal
        non_zero_indices = np.nonzero(cropped_signal)[0]
        damping_info["Interp. frac."] = 1 - len(non_zero_indices)/len(cropped_signal)
        if damping_info["Interp. frac."] > self.settings.max_interp_fraction:
            if self.settings.skip_storing_uncertain_results:
                return None
            damping_info["Note"] += "High interp. frac. "

        # Get the corresponding non-zero values and their indices
        non_zero_values = [cropped_signal[i] for i in non_zero_indices]

        # Generate the new signal with interpolated values
        new_signal = []
        for i, value in enumerate(cropped_signal):
            if not value:
                new_signal.append(float(np.interp(i, non_zero_indices, non_zero_values)))
            else:
                new_signal.append(value)
        return new_signal

    def analyze_frequency_band(self, amp_curve, bottom_row, top_row):
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
        oscillation_info = {
            "Warning": "",
            "Freq. start": self.hht.freq_axis[bottom_row],
            "Freq. stop": self.hht.freq_axis[top_row],
            "Start time": 0.0,  # Start and end time are set in the interpolate_signal function
            "End time": 0.0,
            "NZF": non_zero_fraction,
            "Init. amp.": 0.0,
            "Final amp.": 0.0,
            "Init. amp est.": 0.0,
            "Decay rate": 0.0,
            "Damping ratio": 0.0,
            "Interp. frac.": 0.0,
            "CV": 0.0,
            "Note": ""
        }

        interp_amp_curve = self.interpolate_signal(amp_curve, oscillation_info)

        # If too high interpolated fraction and refusing to store (interpolate_signal() returns None)
        if not interp_amp_curve:
            return

        # Curve fit to find approximate initial amplitude and decay rate
        time_points = np.linspace(0, len(interp_amp_curve)/self.settings.fs, len(interp_amp_curve))
        popt, _ = 0, 0
        try:
            popt, _ = curve_fit(exponential_decay_model, time_points, interp_amp_curve, maxfev=1500)
        except RuntimeError:
            oscillation_info["Note"] += "Skipped. Could not fit curve. "
            self.oscillation_info_list.append(oscillation_info)
            return
        A, decay_rate = popt[0], popt[1]

        #plt.figure()
        #plt.plot(np.arange(oscillation_info["Start time"], oscillation_info["End time"], 1/self.settings.fs), interp_amp_curve, label="Amplitude curve w. interpolation")
        #plt.plot(np.arange(oscillation_info["Start time"], oscillation_info["End time"], 1/self.settings.fs), exponential_decay_model(time_points, A, decay_rate), label="Fitted curve")
        #plt.xlabel("Time [s]")
        #plt.ylabel("Amplitude")
        #plt.legend()
        #plt.tight_layout()

        oscillation_info["CV"]\
            = np.std(interp_amp_curve - exponential_decay_model(time_points, A, decay_rate))/np.mean(interp_amp_curve)
        if oscillation_info["CV"] > self.settings.max_coefficient_of_variation:
            if self.settings.skip_storing_uncertain_results:
                return
            oscillation_info["Note"] += "High CV. "

        oscillation_info["Init. amp."] = interp_amp_curve[0]
        oscillation_info["Final amp."] = interp_amp_curve[-1]
        oscillation_info["Init. amp. est."] = A
        oscillation_info["Decay rate"] = decay_rate

        central_row_ind = (bottom_row + top_row)//2
        oscillation_info["Damping ratio"] = decay_rate/(np.sqrt(decay_rate**2
                                                                + (2*np.pi*self.hht.freq_axis[central_row_ind])**2))

        if oscillation_info["Damping ratio"] < 0:
            if self.settings.segment_length_time - oscillation_info["End time"] <= self.settings.oscillation_timeout:
                oscillation_info["Warning"] = "Negative"
            else:
                oscillation_info["Warning"] = "Ended early"
        elif oscillation_info["Damping ratio"] <= self.settings.damping_ratio_strong_warning_threshold:
            if self.settings.segment_length_time - oscillation_info["End time"] <= self.settings.oscillation_timeout:
                oscillation_info["Warning"] = "Strong"
            else:
                oscillation_info["Warning"] = "Ended early"
        elif oscillation_info["Damping ratio"] <= self.settings.damping_ratio_weak_warning_threshold:
            if self.settings.segment_length_time - oscillation_info["End time"] <= self.settings.oscillation_timeout:
                oscillation_info["Warning"] = "Weak"
            else:
                oscillation_info["Warning"] = "Ended early"

        self.oscillation_info_list.append(oscillation_info)
        return


    def damping_analysis(self):
        """
        Analyzes the damping of different components of the signal segment by performing HHT and using the developed
        algorithm to analyze the Hilbert spectrum.
        :return: None
        """
        start_time = time()
        self.hht.full_hht()  # Calculate hilbert_spectrum and freq_axis
        #self.hht.hilbert_spectrum = self.hht.hilbert_spectrum[::-1]
        #self.hht.freq_axis = self.hht.freq_axis[::-1]

        # Main loop
        n = 0
        while n < len(self.hht.freq_axis):
            # Row adding loop (top)
            # k = 0
            non_zero_count = np.count_nonzero(self.hht.hilbert_spectrum[n])
            if not non_zero_count:
                n += 1
                continue
            combined_row = self.hht.hilbert_spectrum[n]
            combined_row_old = np.copy(combined_row)
            non_zero_count_old = non_zero_count
            k = 1
            while n + k < len(self.hht.freq_axis):
                combined_row_old = np.copy(combined_row)
                non_zero_count_old = non_zero_count
                combined_row = np.maximum(combined_row, self.hht.hilbert_spectrum[n+k])
                non_zero_count = np.count_nonzero(combined_row)
                if non_zero_count - non_zero_count_old > self.settings.minimum_non_zero_improvement \
                    or np.count_nonzero(self.hht.hilbert_spectrum[n+k]) \
                        > self.settings.minimum_total_non_zero_fraction*len(self.input_signal):
                    k += 1
                else:
                    break

            # Row remove loop (bottom)
            m = 1
            while m < k:
                combined_row_old = self.combine_rows(n+m-1, n+k-1)
                non_zero_count_old = np.count_nonzero(combined_row_old)
                combined_row = self.combine_rows(n+m, n+k-1)
                non_zero_count = np.count_nonzero(combined_row)

                row = self.hht.hilbert_spectrum[n+m-1]
                non_zero_count_current = np.count_nonzero(row)
                freq = self.hht.freq_axis[n+m-1]

                if non_zero_count_old - non_zero_count > self.settings.minimum_non_zero_improvement\
                        or np.count_nonzero(self.hht.hilbert_spectrum[n + m])\
                        > self.settings.minimum_total_non_zero_fraction*len(self.input_signal):
                    break
                else:
                    m += 1

            # Calculate and store info if non-zero fraction is large enough
            if non_zero_count_old/len(self.input_signal) > self.settings.minimum_total_non_zero_fraction:
                self.analyze_frequency_band(combined_row_old, n+m-1, n+k-1)
            n += k

        self.runtime = time() - start_time
        if self.settings.print_segment_analysis_time:
            print(f"Segment analysis completed in {self.runtime:.3f} seconds.")

    def combine_rows(self, bottom_row, top_row):
        combined_row = self.hht.hilbert_spectrum[bottom_row]
        if bottom_row == top_row:
            return combined_row

        for i in range(1, top_row - bottom_row + 1):
            combined_row = np.maximum(combined_row, self.hht.hilbert_spectrum[bottom_row + i])
        return combined_row



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