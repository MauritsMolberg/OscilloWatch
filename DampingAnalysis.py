import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from HHT import HHT
from EMD import EMD
from AnalysisSettings import AnalysisSettings


class DampingAnalysis:

    def __init__(self, input_signal, settings: AnalysisSettings):
        self.input_signal = input_signal
        self.settings = settings

        self.hht = HHT(self.input_signal, self.settings)
        self.hht.full_hht()  # Calculate hilbert_spectrum and freq_axis

        self.damping_info_list = []

    def interpolate_signal(self, signal, damping_info):
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
        non_zero_fraction = np.count_nonzero(amp_curve)/len(amp_curve)
        damping_info = {
            "frequency_range": (self.hht.freq_axis[n], self.hht.freq_axis[n+k]),
            "start_time": 0.0,  # Start and end time are set in the interpolate_signal function
            "end_time": 0.0,
            "initial_amplitude": 0.0,
            "decay_constant": 0.0,
            "damping_ratio": 0.0,
            "non_zero_fraction": non_zero_fraction,
            "standard_deviation": 0.0
        }
        interp_amp_curve = self.interpolate_signal(amp_curve, damping_info)

        # Curve fit to find approximate initial amplitude and decay rate
        time_points = np.linspace(0, len(interp_amp_curve)/self.settings.fs, len(interp_amp_curve))
        popt, _ = curve_fit(exponential_decay_model, time_points, interp_amp_curve)
        A, k = popt[0], popt[1]

        damping_info["initial_amplitude"] = A
        row_ind = n + int((k+1)//2)
        damping_info["decay_constant"] = k
        damping_info["damping_ratio"] = k/(np.sqrt(k**2 + (2*np.pi*self.hht.freq_axis[row_ind])**2))
        damping_info["standard_deviation"] = np.std(interp_amp_curve - exponential_decay_model(time_points, A, k))
        print(damping_info, "\n")
        fig, ax = plt.subplots()
        ax.plot(interp_amp_curve)
        ax.plot(exponential_decay_model(time_points, A, k))
        self.damping_info_list.append(damping_info)
        return


    def damping_analysis(self):
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
                        > self.settings.minimum_non_zero_fraction*len(self.input_signal):
                    k += 1
                else:
                    non_zero_count = non_zero_count_old
                    break
            if non_zero_count/len(self.input_signal) > self.settings.minimum_non_zero_fraction:
                self.analyze_frequency_band(combined_row, n, k-1)
            n += k

def remove_short_consecutive_sequences(non_zero_indices, min_consecutive_length):
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
    return A * np.exp(-k * t)



if __name__ == "__main__":
    np.random.seed(0)
    def f(t):
        #return 6*np.exp(.2*t)*np.cos(3*np.pi*t) + 15*np.exp(-.1*t)*np.cos(np.pi*t)
        return (10*np.exp(.2*t)*np.cos(2.4*np.pi*t)
                + 8*np.exp(-.1*t)*np.cos(np.pi*t)
                #+ 3*np.exp(.3*t)*np.cos(5*np.pi*t)
                + 20*np.exp(-.2*t)*np.cos(10*np.pi*t))


    def g(t):
        return 4*np.exp(.2*t)*np.cos(3*np.pi*t)


    settings = AnalysisSettings(remove_padding_after_emd=True, max_imfs=3)
    start = 0
    end = 10
    fs = 50
    t = np.arange(start, end, 1/fs)
    input_signal1 = f(t)
    #input_signal1 = np.load("k2a_with_controls.npy")


    # Random (reproducable) signal
    input_signal2 = np.random.randn(500)

    emd1 = EMD(input_signal1, settings)
    emd1.perform_emd()
    emd1.plot_emd_results(show=False)

    settings.remove_padding_after_emd=False

    hht = HHT(input_signal1, settings)
    hht.full_hht()
    hht.plot_hilbert_spectrum(show=False)

    settings.print_emd_time = True
    settings.print_hht_time = True
    damp = DampingAnalysis(input_signal1, settings)
    damp.damping_analysis()

    plt.show()

    """
    def f(t):
        return 5*np.exp(-.5*t)

    start = 0
    end = 10
    fs = 50
    t = np.arange(start, end, 1/fs)
    input_signal1 = 5*np.exp(-.5*t)
    input_signal2 = 8*np.exp(-.9*t)
    input_signal3 = 2*np.exp(.3*t)


    input_signal1[55:70] = 0.0
    input_signal1[100:130] = 0.0
    input_signal1[350:400] = 0.0

    hilbert_spectrum = np.array([
        np.zeros((end - start)*fs),
        np.zeros((end - start)*fs),
        input_signal1,
        np.zeros((end - start)*fs),
        np.concatenate((input_signal2[:440], np.zeros(60))),
        input_signal3
    ])

    freq_axis = np.arange(len(hilbert_spectrum))

    settings = AnalysisSettings()
    damp = DampingAnalysis(input_signal1, settings)


    damp.damping_analysis(hilbert_spectrum, freq_axis)

    """


