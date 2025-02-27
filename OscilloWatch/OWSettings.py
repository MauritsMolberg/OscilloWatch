#################################################################################
#     OscilloWatch, real-time power system oscillation detection tool.
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

class OWSettings:

    def __init__(self,

                 # General settings
                 fs=50,

                 # Segment analysis settings
                 segment_length_time=10.0,
                 mirror_padding_fraction=1.0,
                 extension_padding_time_start=0.0,
                 extension_padding_time_end=0.0,
                 remove_padding_after_emd=False,
                 remove_padding_after_hht=True,

                 # EMD settings
                 emd_rec_tolerance=0.003,
                 max_imfs=5,
                 max_emd_sifting_iterations=100,
                 emd_min_peaks=4,
                 emd_add_edges_to_peaks=True,

                 # HHT settings
                 minimum_amplitude=0,
                 hht_frequency_resolution=None,  # Default: Equal to inverse of sampling rate
                 hht_amplitude_moving_avg_window=5,
                 hht_frequency_moving_avg_window=41,
                 hht_split_signal_freq_change_enable=True,
                 hht_split_signal_freq_change_threshold=0.5,
                 hht_split_signal_freq_change_nperseg=100,
                 minimum_frequency=None,  # Default: Uses Nyquist–Shannon sampling theorem (1/segment length)
                 maximum_frequency=None,  # Default: Unbounded

                 # Hilbert spectrum analysis settings
                 minimum_total_non_zero_fraction=0.1,
                 minimum_consecutive_non_zero_length=5,
                 minimum_non_zero_improvement=3,
                 max_coefficient_of_variation=0.4,
                 max_interp_fraction=0.4,
                 skip_storing_uncertain_modes=False,
                 start_amp_curve_at_peak=True,

                 # Alarm settings
                 alarm_median_amplitude_threshold=0.0,
                 damping_ratio_weak_alarm_threshold=0.15,
                 damping_ratio_strong_alarm_threshold=0.05,
                 oscillation_timeout=2.0,
                 segment_memory_freq_threshold=0.15,

                 # Result format settings
                 results_file_path="results/results",
                 store_csv=True,
                 store_pkl=True,
                 csv_decimals=5,
                 csv_delimiter=",",
                 unit=None,
                 include_advanced_results=False,
                 include_asterisk_explanations=True,

                 # Print settings
                 print_segment_number=True,
                 print_emd_time=False,
                 print_hht_time=False,
                 print_segment_analysis_time=True,
                 print_alarms=True,
                 print_emd_sifting_details=False,

                 # Real-time analysis settings
                 ip="localhost",
                 port=50000,
                 sender_device_id=45,
                 pmu_id=3000,
                 channel="V_1",
                 phasor_component="magnitude"
                 ):
        """
        Constructs OWSettings object, which is used for all OscilloWatch functionality.

        :param int fs: Sampling frequency in hertz.
        :param float | int segment_length_time: Length of each segment in seconds.
        :param float | int mirror_padding_fraction: Fraction of the signal that is mirrored at the beginning and end of
         the segment.
        :param float | int extension_padding_time_start: Length of the signal from before the segment that is used as
         padding.
        :param float | int extension_padding_time_end: Length of the signal from after the segment that is used as
         padding.
        :param bool remove_padding_after_emd: Removes padding by the end of EMD if True, and keeps it if False (includes
         for HHT).
        :param bool remove_padding_after_hht: Removes padding by the end of HHT if True, and keeps it if False. No
         effect if remove_padding_after_emd is True.
        :param float emd_rec_tolerance: EMD sifting stops if REC between the last two iterations is smaller than this
         value.
        :param int max_imfs: Maximum number of IMFs before the EMD is stopped.
        :param int max_emd_sifting_iterations: Maximum number of sifting iterations before EMD sifting is stopped.
        :param int emd_min_peaks: Minimum number of both upper and lower peaks in the residual to continue the EMD
         process. If there are fewer upper or lower peaks than this number, the EMD process is stopped. The
         interpolation needs at least 2 points, so if this minimum is set lower than 2, it will automatically be
         increased to 2.
        :param bool emd_add_edges_to_peaks: Both edge points are included in the lists of upper and lower peaks if True.
        :param float minimum_amplitude: Minimum amplitude of a time-frequency point to not be removed from the Hilbert
         spectrum. Should be set higher than the amplitude of the noise to filter it out.
        :param float | None hht_frequency_resolution: Distance between each value in the discrete frequency axis in the
         Hilbert spectrum. If None (default), will be the same as the inverse of the sampling rate (1:1 scale of time
         and frequency axes).
        :param int hht_amplitude_moving_avg_window: Window size used for the moving average filter applied to the
         instantaneous amplitude curves, measured in number of samples.
        :param int hht_frequency_moving_avg_window: Window size used for the moving average filter applied to the
         instantaneous frequency curves, measured in number of samples.
        :param bool hht_split_signal_freq_change_enable: Enables splitting an IMF before Hilbert transforming if its
         dominating frequency changes abruptly if True.
        :param float hht_split_signal_freq_change_threshold: Minimum amplitude of spike in gradient of dominating
         frequency for the IMF to be split at the spike before Hilbert transform. No effect if hht split signal freq
         change toggle is False.
        :param int hht_split_signal_freq_change_nperseg: Window width of STFT measured in number of samples used for
         deciding where to split IMFs. No effect if hht_split_signal_freq_change_toggle is False.
        :param float | None minimum_frequency: Lower bound of frequency axis in Hilbert spectrum. If None (default),
         the inverse of the length of the segment plus extension padding will be used (Nyquist-Shannon sampling
         theorem.)
        :param float | None maximum_frequency: Upper bound of frequency axis in Hilbert spectrum. If None (default),
         there is no upper bound, and the highest frequency found in the IMFs will be the highest value in the frequency
         axis.
        :param float minimum_total_non_zero_fraction: Minimum fraction of the combined row from the Hilbert spectrum
         that must be non-zero for the frequency band to be interpreted as a mode and analyzed.
        :param int minimum_consecutive_non_zero_length: Minimum consecutive series of non-zero elements in a combined
         row an element has to be part of to not be set to zero.
        :param int minimum_non_zero_improvement: Minimum improvement to number of non-zero elements in combined row from
         including another row for this new row to be included in the combined row.
        :param float max_coefficient_of_variation: Maximum coefficient of variation between an amplitude curve with
         interpolation and the fitted curve before the estimated damping is considered uncertain.
        :param float max_interp_fraction: Maximum fraction of the row elements that had to be filled with interpolation
         before the mode is considered uncertain.
        :param bool skip_storing_uncertain_modes: Does not store details on modes with interpolation fraction exceeding
         max_interp_fraction if True. Stores the modes with an asterisk and brackets around the frequency in the
         CSV file if False.
        :param bool start_amp_curve_at_peak: Ignores all amplitude values before the maximum value in the row if this
         maximum is in the first half of the row’s non-zero portion if True.
        :param float alarm_median_amplitude_threshold: The minimum median amplitude of the mode in the segment for an
         alarm to be raised.
        :param float damping_ratio_weak_alarm_threshold: Damping ratio threshold for a weak alarm to be raised.
        :param float damping_ratio_strong_alarm_threshold: Damping ratio threshold for a strong alarm to be raised.
        :param float | int oscillation_timeout: Maximum number of seconds before end of segment the detected oscillation
         can end before it is interpreted as ended, and any alarm is suppressed.
        :param float segment_memory_freq_threshold: The tolerance range for frequency in both positive and negative
         directions for two modes in two consecutive segments to be considered the same mode. Used to determine if the
         mode is sustained.
        :param str results_file_path: File path to where results files (CSV and PKL) will be stored, including file
         name, but NOT file extension.
        :param bool store_csv: Enables storing results table to CSV file if True.
        :param bool store_pkl: Enables storing results objects to PKL file if True.
        :param int csv_decimals: Number of decimals used in results table in PKL file.
        :param str csv_delimiter: Delimiter used between values in CSV file. Must be set to ";" instead of "," to open
         in Excel if Excel interprets "," as decimal separator (common in most European countries and others).
        :param str | None unit: Unit to be placed in "[]" after amplitudes in header of CSV table. If None (default), no
         unit or "[]" will be included.
        :param bool include_advanced_results: Includes columns for advanced results in CSV if True. Only includes basic
         results if False.
        :param bool include_asterisk_explanations: Includes explanations for asterisks given for uncertain results at
         the top or bottom of CSV file if True. Still includes asterisks, but with no explanations if False.
        :param bool print_segment_number: Prints the segment number that is currently being analyzed to the console if
         True.
        :param bool print_emd_time: Prints the runtime of the EMD step of the segment analysis algorithm to the output
         console for each segment if True.
        :param bool print_hht_time: Prints the runtime of the whole HHT step of the segment analysis algorithm to the
         output console, including EMD, for each segment if True.
        :param bool print_segment_analysis_time: Prints the runtime of the whole segment analysis algorithm to the
         output console for each segment if True.
        :param bool print_alarms: Prints notice about alarms including their frequency to the output console when they
         occur.
        :param bool print_emd_sifting_details: Prints the number of sifting iterations for each IMF, along with REC for
         each of them, to the output console if True.
        :param str ip: IP address of sending device to connect to when using real-time analysis mode.
        :param int port: Port number of sending device.
        :param int sender_device_id: ID code of sending device. ("Device ID Code" in PMU Connection Tester.)
        :param int pmu_id: ID code of PMU with the data you want to analyze. ("ID Code" behind "PMU:" in PMU Connection
         Tester.)
        :param str channel: Name of channel you want to connect to. Phasor name if you want to read phasor data, or
         "frequency" or "freq" if you want to analyze the system frequency. ("The part after "V:" or "I:" in the
         "Phasor:" drop-down menu in PMU Connection Tester if you want to analyze a phasor.)
        :param str phasor_component: Component of the phasor you want to analyze. Can be "magnitude" or "angle".
        """

        self.fs = fs
        self.segment_length_time = segment_length_time
        self.print_segment_number = print_segment_number
        self.print_emd_time = print_emd_time
        self.print_hht_time = print_hht_time
        self.print_segment_analysis_time = print_segment_analysis_time
        self.print_alarms = print_alarms

        self.csv_decimals = csv_decimals
        self.csv_delimiter = csv_delimiter
        self.results_file_path = results_file_path

        self.emd_rec_tolerance = emd_rec_tolerance
        self.max_imfs = max_imfs
        self.max_emd_sifting_iterations = max_emd_sifting_iterations
        self.print_emd_sifting_details = print_emd_sifting_details
        self.emd_min_peaks = max(emd_min_peaks, 2)  # Needs to be at least 2
        self.emd_add_edges_to_peaks = emd_add_edges_to_peaks

        self.mirror_padding_fraction = mirror_padding_fraction
        self.extension_padding_time_start = extension_padding_time_start
        self.extension_padding_time_end = extension_padding_time_end
        self.remove_padding_after_emd = remove_padding_after_emd
        self.remove_padding_after_hht = remove_padding_after_hht

        self.minimum_amplitude = minimum_amplitude

        self.hht_amplitude_moving_avg_window = hht_amplitude_moving_avg_window
        self.hht_frequency_moving_avg_window = hht_frequency_moving_avg_window
        self.hht_split_signal_freq_change_enable = hht_split_signal_freq_change_enable
        self.hht_split_signal_freq_change_threshold = hht_split_signal_freq_change_threshold
        self.hht_split_signal_freq_change_nperseg = hht_split_signal_freq_change_nperseg
        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency

        self.store_csv = store_csv
        self.store_pkl = store_pkl
        self.unit = unit
        self.skip_storing_uncertain_modes = skip_storing_uncertain_modes
        self.minimum_total_non_zero_fraction = minimum_total_non_zero_fraction
        self.minimum_consecutive_non_zero_length = minimum_consecutive_non_zero_length
        self.minimum_non_zero_improvement = minimum_non_zero_improvement
        self.max_coefficient_of_variation = max_coefficient_of_variation
        self.max_interp_fraction = max_interp_fraction
        self.start_amp_curve_at_peak = start_amp_curve_at_peak
        self.include_advanced_results = include_advanced_results
        self.include_asterisk_explanations = include_asterisk_explanations

        self.ip = ip
        self.port = port
        self.sender_device_id = sender_device_id
        self.pmu_id = pmu_id
        self.channel = channel
        self.phasor_component = phasor_component

        self.alarm_median_amplitude_threshold = alarm_median_amplitude_threshold
        self.damping_ratio_weak_alarm_threshold = damping_ratio_weak_alarm_threshold
        self.damping_ratio_strong_alarm_threshold = damping_ratio_strong_alarm_threshold
        self.oscillation_timeout = oscillation_timeout
        self.segment_memory_freq_threshold = segment_memory_freq_threshold

        # Initialize variables that will be calculated in update_calc_values
        self.segment_length_samples = 0
        self.extension_padding_samples_start = 0
        self.extension_padding_samples_end = 0
        self.total_segment_length_samples = 0
        self.hht_frequency_resolution = hht_frequency_resolution

        self.update_calc_values()

    def update_calc_values(self):
        """
        Updates values that need to be calculated, based on the current parameter values. Must be run after a parameter
        like the sampling rate is updated.
        :return: None
        """
        self.segment_length_samples = int(self.segment_length_time*self.fs)
        self.extension_padding_samples_start = int(self.extension_padding_time_start*self.fs)
        self.extension_padding_samples_end = int(self.extension_padding_time_end*self.fs)
        self.total_segment_length_samples = (self.segment_length_samples
                                             + self.extension_padding_samples_start
                                             + self.extension_padding_samples_end)

        if self.hht_frequency_resolution is None:
            self.hht_frequency_resolution = 1/self.fs
