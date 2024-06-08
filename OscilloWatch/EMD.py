#################################################################################
#     OscilloWatch, early warning application for low-frequency oscillations
#     Copyright (C) 2024  Maurits Sørensen Molberg
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
#################################################################################

from time import time

from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np

from OscilloWatch.OWSettings import OWSettings


class EMD:
    """
    Class for performing Empirical Mode Decomposition on a signal segment.
    """

    def __init__(self, input_signal, settings: OWSettings):
        """
        Constructor for EMD class. Initializes variables, but does not perform the actual EMD algorithm.

        :param input_signal: Signal to be decomposed.
        :type input_signal: numpy.ndarray or list
        :param settings: Object containing the settings for the EMD algorithm, and the other OscilloWatch that will be used
            in the signal analysis
        :type settings: OWSettings
        """
        self.settings = settings
        self.input_signal = input_signal
        self.extended_signal = np.array([])

        self.imf_list = []
        self.residual = np.array([])

        self.runtime = 0

    def get_envelopes(self, signal):
        """
        Calculates and returns the peaks of the input signal and constructs the upper and lower envelopes using cubic spline
        interpolation.

        :param signal: Signal for which the envelopes should be calculated.
        :type signal: numpy.ndarray or list

        :return upper_envelope: Envelope interpolated using upper peaks.
        :rtype upper_envelope: numpy.ndarray
        :return lower_envelope: Envelope interpolated using lower peaks.
        :rtype lower_envelope: numpy.ndarray
        """
        upper_peaks, _ = find_peaks(signal)
        lower_peaks, _ = find_peaks(-signal)

        # Stop criteria
        if len(upper_peaks) < self.settings.emd_min_peaks or len(lower_peaks) < self.settings.emd_min_peaks:
            return None, None

        if self.settings.emd_add_edges_to_peaks:
            upper_peaks = np.insert(upper_peaks, 0, 0)
            upper_peaks = np.append(upper_peaks, len(signal) - 1)
            lower_peaks = np.insert(lower_peaks, 0, 0)
            lower_peaks = np.append(lower_peaks, len(signal) - 1)

        xAxis = np.arange(len(signal))
        interp_upper = CubicSpline(upper_peaks, signal[upper_peaks])
        interp_lower = CubicSpline(lower_peaks, signal[lower_peaks])

        upper_envelope = interp_upper(xAxis)
        lower_envelope = interp_lower(xAxis)

        return upper_envelope, lower_envelope

    def perform_emd(self):
        """
        Performs the EMD algorithm on the input signal of the EMD object. Updates object's imf_list and residual
        variables.

        :return: None
        """
        start_time = time()

        r = np.copy(self.input_signal) # Updated every time an IMF is subtracted from it
        original_length = len(self.input_signal)

        # Mirror padding, to handle boundary effects
        if self.settings.mirror_padding_fraction > 0: # Skip if zero or negative
            num_samples_to_mirror = int(original_length*self.settings.mirror_padding_fraction)

            mirrored_fraction_start = r[:num_samples_to_mirror][::-1]
            mirrored_fraction_end = r[-num_samples_to_mirror:][::-1]
            self.extended_signal = np.concatenate((mirrored_fraction_start, r, mirrored_fraction_end))
            r = np.copy(self.extended_signal)

        # Set to True if no more IMFs can be extracted, i.e. if residual has less than 4 upper and lower peaks.
        # Breaks out of both for-loops if True.
        emd_done = False

        # EMD algorithm start
        for n in range(self.settings.max_imfs):
            h = np.copy(r)

            if self.settings.print_emd_sifting_details:
                print("----------------\nIMF iteration:", n+1)

            # Sifting process start
            for k in range(self.settings.max_emd_sifting_iterations):
                upper_envelope, lower_envelope = self.get_envelopes(h)

                if upper_envelope is None or lower_envelope is None:  # h has less than 4 upper and lower peaks
                    emd_done = True  # Breaks out of both loops
                    break

                avg_envelope = (upper_envelope + lower_envelope) / 2
                h_old = np.copy(h)
                h -= avg_envelope

                #std = np.std(h-h_old, dtype=np.float64)
                energy_old = np.sum(h_old**2)
                energy_new = np.sum(h**2)
                rec = (energy_old - energy_new) / energy_old
                if self.settings.print_emd_sifting_details:
                    print("Sifting iteration:", k+1, "\nREC:", rec)

                if abs(rec) < self.settings.emd_rec_tolerance:
                    break

            if emd_done:
                break

            self.imf_list.append(h)
            r -= h

        # Removing the extensions added for padding if desired
        if self.settings.remove_padding_after_emd:
            ext_len = int(original_length * self.settings.mirror_padding_fraction)
            self.imf_list = [imf[ext_len + self.settings.extension_padding_samples_start
                                 :-ext_len - self.settings.extension_padding_samples_end - 1] for imf in self.imf_list]
            r = r[ext_len + self.settings.extension_padding_samples_start
                  :-ext_len - self.settings.extension_padding_samples_end - 1]

        self.residual = r

        self.runtime = time() - start_time
        if self.settings.print_emd_time:
            print(f"EMD completed in {self.runtime:.3f} seconds.")
        return

    def plot_emd_results(self, show = False, include_padding = False):
        """
        Plots input signal and IMFs of EMD object. The function perform_emd() must have been run before.

        :param bool show: Specifies whether the plt.show() should be run at the end of the function.
        :param bool include_padding: Includes all padding in the plots if True. Removes them first if False. No effect
            if the remove_padding_after_emd setting is True.
        :return: None
        """
        if self.settings.remove_padding_after_emd:  # Do not remove padding again if already removed
            orig_signal = self.input_signal[self.settings.extension_padding_samples_start
                                            :-self.settings.extension_padding_samples_end - 1:]
            new_imf_list = self.imf_list
            res_new = self.residual
        elif include_padding:
            orig_signal = self.extended_signal
            new_imf_list = self.imf_list
            res_new = self.residual
        else:
            orig_signal = self.input_signal[self.settings.extension_padding_samples_start
                                            :-self.settings.extension_padding_samples_end - 1:]
            ext_len = int(len(self.input_signal)*self.settings.mirror_padding_fraction)
            new_imf_list = [imf[ext_len + self.settings.extension_padding_samples_start
                                :-ext_len - self.settings.extension_padding_samples_end - 1] for imf in self.imf_list]
            res_new = self.residual[ext_len + self.settings.extension_padding_samples_start
                                :-ext_len - self.settings.extension_padding_samples_end - 1]

        tAxis = np.linspace(0, len(orig_signal)/self.settings.fs, len(orig_signal))

        num_imfs = len(self.imf_list)

        # Create a grid of subplots for IMFs and residual
        fig, axes = plt.subplots(num_imfs + 2, 1, figsize=(5, 1.5*(num_imfs + .7)), sharex=True)

        # Plot the input signal
        axes[0].plot(tAxis, orig_signal, color='blue', linewidth = .7)
        axes[0].set_title('Input Signal')

        # Plot each IMF
        for i, imf in enumerate(new_imf_list):
            axes[i + 1].plot(tAxis, imf, color='green', linewidth = .7)
            axes[i + 1].set_title(f'IMF {i + 1}')

        # Plot the residual
        axes[num_imfs+1].plot(tAxis, res_new, color='red', linewidth = .7)
        axes[num_imfs+1].set_title('Residual')

        axes[num_imfs+1].set_xlabel('Time [s]')

        plt.tight_layout()
        if show:
            plt.show()
