class AnalysisSettings:

    def __init__(self,
                 fs=50,
                 noise_reduction_moving_avg_window = 1,

                 emd_sd_tolerance=0.2,
                 max_imfs=10,
                 max_emd_sifting_iterations=30,
                 mirror_padding_fraction=1,
                 print_emd_sifting_details=False,
                 print_emd_time=False,
                 print_hht_time=False,

                 remove_padding_after_emd=False,
                 remove_padding_after_hht=True,
                 hht_amplitude_threshold=0,
                 hht_frequency_resolution="auto",
                 hht_frequency_threshold="auto",
                 hht_frequency_spike_threshold=0.7,
                 hht_max_frequency_spike_duration=5,
                 hht_amplitude_moving_avg_window=5,
                 hht_frequency_moving_avg_window=21,
                 hht_split_signal_freq_change_toggle=True,
                 hht_split_signal_freq_change_threshold=0.5,
                 hht_split_signal_freq_change_nperseg=100,

                 minimum_non_zero_fraction=0.2,
                 minimum_consecutive_non_zero_length=5,
                 minimum_non_zero_improvement=4,
                 max_freq_band_width=1.5
                 ):

        self.fs = fs
        self.noise_reduction_moving_avg_window = noise_reduction_moving_avg_window

        self.emd_sd_tolerance = emd_sd_tolerance
        self.max_imfs = max_imfs
        self.max_emd_sifting_iterations = max_emd_sifting_iterations
        self.mirror_padding_fraction = mirror_padding_fraction
        self.print_emd_sifting_details = print_emd_sifting_details
        self.print_emd_time = print_emd_time
        self.print_hht_time = print_hht_time

        self.remove_padding_after_emd = remove_padding_after_emd
        self.remove_padding_after_hht = remove_padding_after_hht
        self.hht_amplitude_threshold = hht_amplitude_threshold
        self.hht_frequency_resolution = hht_frequency_resolution
        self.hht_frequency_threshold = hht_frequency_threshold
        self.hht_frequency_spike_threshold = hht_frequency_spike_threshold
        self.hht_max_frequency_spike_duration = hht_max_frequency_spike_duration
        self.hht_amplitude_moving_avg_window = hht_amplitude_moving_avg_window
        self.hht_frequency_moving_avg_window = hht_frequency_moving_avg_window
        self.hht_split_signal_freq_change_toggle = hht_split_signal_freq_change_toggle
        self.hht_split_signal_freq_change_threshold = hht_split_signal_freq_change_threshold
        self.hht_split_signal_freq_change_nperseg = hht_split_signal_freq_change_nperseg

        self.minimum_non_zero_fraction = minimum_non_zero_fraction
        self.minimum_consecutive_non_zero_length = minimum_consecutive_non_zero_length
        self.minimum_non_zero_improvement = minimum_non_zero_improvement
        self.max_freq_band_width = max_freq_band_width