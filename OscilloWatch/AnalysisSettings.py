class AnalysisSettings:

    def __init__(self,

                 # General settings
                 fs=50,
                 print_segment_number=True,
                 print_emd_time=False,
                 print_hht_time=False,
                 print_segment_analysis_time=False,
                 print_alarms=False,

                 # File storing settings
                 csv_decimals=5,
                 csv_delimiter=";",
                 results_file_path="../results/results",

                 # EMD settings
                 emd_rec_tolerance=0.003,
                 max_imfs=10,
                 max_emd_sifting_iterations=100,
                 print_emd_sifting_details=False,
                 emd_min_peaks=4,
                 emd_add_edges_to_peaks=False,

                 # EMD/HHT settings
                 mirror_padding_fraction=1,
                 extension_padding_time_start=0,
                 extension_padding_time_end=0,
                 remove_padding_after_emd=False,
                 remove_padding_after_hht=True,

                 # HHT settings
                 hht_amplitude_threshold=1e-6,
                 hht_frequency_resolution=None,
                 hht_frequency_threshold=None,
                 hht_amplitude_moving_avg_window=5,
                 hht_frequency_moving_avg_window=41,
                 hht_split_signal_freq_change_toggle=True,
                 hht_split_signal_freq_change_threshold=0.5,
                 hht_split_signal_freq_change_nperseg=100,
                 min_freq = None,
                 max_freq = None,

                 # Result format settings
                 skip_storing_uncertain_modes=False,
                 minimum_total_non_zero_fraction=0.1,
                 minimum_consecutive_non_zero_length=5,
                 minimum_non_zero_improvement=3,
                 max_coefficient_of_variation=0.4,
                 max_interp_fraction=0.4,
                 include_advanced_results=False,

                 # Mode characterization settings
                 start_amp_curve_at_peak=True,

                 # Segment analysis settings
                 segment_length_time=10,

                 # Real-time analysis settings
                 ip="localhost",
                 port=50000,
                 sender_device_id=45,
                 pmu_id=3000,
                 channel="V_1",
                 phasor_component="magnitude",

                 # Alarm settings
                 damping_ratio_weak_alarm_threshold=0.15,
                 damping_ratio_strong_alarm_threshold=0.05,
                 oscillation_timeout=2,
                 segment_memory_freq_threshold=0.1
                 ):

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

        self.hht_amplitude_threshold = hht_amplitude_threshold

        self.hht_amplitude_moving_avg_window = hht_amplitude_moving_avg_window
        self.hht_frequency_moving_avg_window = hht_frequency_moving_avg_window
        self.hht_split_signal_freq_change_toggle = hht_split_signal_freq_change_toggle
        self.hht_split_signal_freq_change_threshold = hht_split_signal_freq_change_threshold
        self.hht_split_signal_freq_change_nperseg = hht_split_signal_freq_change_nperseg
        self.min_freq = min_freq
        self.max_freq = max_freq

        self.skip_storing_uncertain_modes = skip_storing_uncertain_modes
        self.minimum_total_non_zero_fraction = minimum_total_non_zero_fraction
        self.minimum_consecutive_non_zero_length = minimum_consecutive_non_zero_length
        self.minimum_non_zero_improvement = minimum_non_zero_improvement
        self.max_coefficient_of_variation = max_coefficient_of_variation
        self.max_interp_fraction = max_interp_fraction
        self.start_amp_curve_at_peak = start_amp_curve_at_peak
        self.include_advanced_results = include_advanced_results

        self.ip = ip
        self.port = port
        self.sender_device_id = sender_device_id
        self.pmu_id = pmu_id
        self.channel = channel
        self.phasor_component = phasor_component

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
        self.hht_frequency_threshold = hht_frequency_threshold

        self.update_calc_values()

        self.blank_mode_info_dict = {  # Practical to have here, so the keys can be easily fetched from anywhere
            "Mode status": "",
            "Damping evaluation": "",
            "Alarm": "",
            "Frequency": 0.0,
            "Damping ratio": 0.0,
            "Init. amp.": 0.0,
            "Final amp.": 0.0,
            "Start time": 0.0,
            "End time": 0.0,
            "Freq. start": 0.0,
            "Freq. stop": 0.0,
            "Init. amp. est.": 0.0,
            "Decay rate": 0.0,
            "NZF": 0.0,
            "Interp. frac.": 0.0,
            "CV": 0.0,
            "Note": ""
        }

        self.blank_mode_info_dict_simple = {
            "Mode status": "",
            "Damping evaluation": "",
            "Alarm": "",
            "Frequency": 0.0,
            "Damping ratio": 0.0,
            "Init. amp.": 0.0,
            "Final amp.": 0.0,
            "Start time": 0.0,
            "End time": 0.0,
            "Note": ""
        }

    def update_calc_values(self):
        self.segment_length_samples = int(self.segment_length_time*self.fs)
        self.extension_padding_samples_start = int(self.extension_padding_time_start*self.fs)
        self.extension_padding_samples_end = int(self.extension_padding_time_end*self.fs)
        self.total_segment_length_samples = (self.segment_length_samples
                                             + self.extension_padding_samples_start
                                             + self.extension_padding_samples_end)

        if self.hht_frequency_resolution is None:
            self.hht_frequency_resolution = 1/self.fs
        if self.hht_frequency_threshold is None:
            self.hht_frequency_threshold = self.hht_frequency_resolution/2


