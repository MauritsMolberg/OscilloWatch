class AnalysisSettings:

    def __init__(self,

                 # General settings
                 fs=50,
                 noise_reduction_moving_avg_window=1,
                 print_segment_number=False,
                 print_emd_time=False,
                 print_hht_time=False,
                 print_segment_analysis_time=False,
                 csv_decimals=5,
                 csv_delimiter=";",
                 max_csv_save_attempts=9,

                 # EMD settings
                 emd_rec_tolerance=0.003,
                 max_imfs=10,
                 max_emd_sifting_iterations=100,
                 print_emd_sifting_details=False,
                 emd_min_peaks=4,

                 # EMD/HHT settings
                 mirror_padding_fraction=1,
                 extension_padding_time_start=0,
                 extension_padding_time_end=0,
                 remove_padding_after_emd=False,
                 remove_padding_after_hht=True,

                 # HHT settings
                 hht_amplitude_threshold=1e-6,
                 hht_frequency_resolution="auto",
                 hht_frequency_threshold="auto",
                 hht_amplitude_moving_avg_window=5,
                 hht_frequency_moving_avg_window=41,
                 hht_split_signal_freq_change_toggle=True,
                 hht_split_signal_freq_change_threshold=0.5,
                 hht_split_signal_freq_change_nperseg=100,

                 # Result exclusion settings
                 skip_storing_uncertain_results=False,
                 minimum_total_non_zero_fraction=0.1,
                 minimum_consecutive_non_zero_length=5,
                 minimum_non_zero_improvement=3,
                 max_coefficient_of_variation=0.4,
                 max_interp_fraction=0.4,
                 start_amp_curve_at_peak=True,

                 # Segment analysis settings
                 segment_length_time=10,

                 # Real time analysis settings
                 pmu_ip = "localhost",
                 pmu_port = 50000,
                 pdc_id =1410,

                 # Warning settings
                 damping_ratio_weak_warning_threshold=0.15,
                 damping_ratio_strong_warning_threshold=0.05,
                 oscillation_timeout=2

                 ):

        self.fs = fs
        self.segment_length_time = segment_length_time
        self.segment_length_samples = segment_length_time*fs
        self.noise_reduction_moving_avg_window = noise_reduction_moving_avg_window
        self.print_segment_number = print_segment_number
        self.print_emd_time = print_emd_time
        self.print_hht_time = print_hht_time
        self.print_segment_analysis_time = print_segment_analysis_time
        self.csv_decimals = csv_decimals
        self.csv_delimiter = csv_delimiter
        self.max_csv_save_attempts = max_csv_save_attempts

        self.emd_rec_tolerance = emd_rec_tolerance
        self.max_imfs = max_imfs
        self.max_emd_sifting_iterations = max_emd_sifting_iterations
        self.print_emd_sifting_details = print_emd_sifting_details
        self.emd_min_peaks = max(emd_min_peaks, 2)  # Needs to be at least 2

        self.mirror_padding_fraction = mirror_padding_fraction
        self.extension_padding_time_start = extension_padding_time_start
        self.extension_padding_time_end = extension_padding_time_end
        self.extension_padding_samples_start = extension_padding_time_start*fs
        self.extension_padding_samples_end = extension_padding_time_end*fs
        self.total_segment_length_samples = (self.segment_length_samples
                                             + self.extension_padding_samples_start
                                             + self.extension_padding_samples_end)
        self.remove_padding_after_emd = remove_padding_after_emd
        self.remove_padding_after_hht = remove_padding_after_hht

        self.hht_amplitude_threshold = hht_amplitude_threshold

        if hht_frequency_resolution == "auto":
            self.hht_frequency_resolution = 1/self.fs
        else:
            self.hht_frequency_resolution = hht_frequency_resolution

        if hht_frequency_threshold == "auto":
            self.hht_frequency_threshold = self.hht_frequency_resolution/2
        else:
            self.hht_frequency_threshold = hht_frequency_threshold

        self.hht_amplitude_moving_avg_window = hht_amplitude_moving_avg_window
        self.hht_frequency_moving_avg_window = hht_frequency_moving_avg_window
        self.hht_split_signal_freq_change_toggle = hht_split_signal_freq_change_toggle
        self.hht_split_signal_freq_change_threshold = hht_split_signal_freq_change_threshold
        self.hht_split_signal_freq_change_nperseg = hht_split_signal_freq_change_nperseg

        self.skip_storing_uncertain_results = skip_storing_uncertain_results
        self.minimum_total_non_zero_fraction = minimum_total_non_zero_fraction
        self.minimum_consecutive_non_zero_length = minimum_consecutive_non_zero_length
        self.minimum_non_zero_improvement = minimum_non_zero_improvement
        self.max_coefficient_of_variation = max_coefficient_of_variation
        self.max_interp_fraction = max_interp_fraction
        self.start_amp_curve_at_peak = start_amp_curve_at_peak

        self.pmu_ip = pmu_ip
        self.pmu_port = pmu_port
        self.pdc_id = pdc_id

        self.damping_ratio_weak_warning_threshold = damping_ratio_weak_warning_threshold
        self.damping_ratio_strong_warning_threshold = damping_ratio_strong_warning_threshold
        self.oscillation_timeout = oscillation_timeout
