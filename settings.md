## Basic settings

### General settings
* `fs` Sampling frequency in hertz. Default: 50.
* `minimum_amplitude` Minimum amplitude of a time-frequency point to not be removed from the Hilbert spectrum. Should be set higher than the amplitude of the noise to filter it out. Default: 0.0.

### Segment analysis settings
* `segment_length_time` Length of each segment in seconds. Default: 10.0.
* `extension_padding_time_start` Length of the signal from before the segment that is used as padding. Default: 0.0.
* `extension_padding_time_end` Length of the signal from after the segment that is used as padding. Default: 0.0.

### Result format settings
* `results_file_path` File path to where results files (CSV and PKL) will be stored, including file name, but ***not*** file extension. Default: "results/results".
* `include_asterisk_explanations` Includes explanations for asterisks given for uncertain results at the top or bottom of CSV file if True. Still includes asterisks, but with no explanations if False. Default: True.

### Real-time analysis settings
* `ip` IP address of sending device to connect to. Default: "localhost".
* `port` Port number of sending device. Default: 50000.
* `sender_device_id` ID code of sending device. ("Device ID Code" in [PMU Connection Tester](https://github.com/GridProtectionAlliance/PMUConnectionTester).) Default: 45.
* `pmu_id` ID code of PMU with the data you want to analyze. ("ID Code" behind "PMU:" in PMU Connection Tester.) Default: 3000.
* `channel` Name of channel you want to connect to. Phasor name if you want to read phasor data, or "frequency" or "freq" if you want to analyze the system frequency. ("The part after "V:" or "I:" in the "Phasor:" drop-down menu in PMU Connection Tester if you want to analyze a phasor.) Default: "V_1".
* `phasor_component` Component of the phasor you want to analyze. Can be "magnitude" or "angle". Default: "magnitude".

---

## Advanced settings

### Segment analysis settings

* `mirror_padding_fraction` Fraction of the signal that is mirrored at the beginning and end of the segment. Default: 1.0.
* `remove_padding_after_emd` Removes padding by the end of EMD if True, and keeps it if False (includes for HHT). Default: False.
* `remove_padding_after_hht` Removes padding by the end of HHT if True, and keeps it if False. No effect if `remove_padding_after_emd` is True. Default: True.


### EMD settings

* `emd_rec_tolerance` EMD sifting stops if REC between the last two iterations is smaller than this value. Default: 0.003.
* `max_imfs` Maximum number of IMFs before the EMD is stopped. Default: 5.
* `max_emd_sifting_iterations` Maximum number of sifting iterations before EMD sifting is stopped. Default: 100.
* `emd_min_peaks` Minimum number of both upper and lower peaks in the residual to continue the EMD process. If there are fewer upper ***or*** lower peaks than this number, the EMD process is stopped. The interpolation needs at least 2 points, so if this minimum is set lower than 2, it will automatically be increased to 2. Default: 4.
* `emd_add_edges_to_peaks` Both edge points are included in the lists of upper and lower peaks if True. Default: True.


### HHT settings

* `hht_frequency_resolution` Distance between each value in the discrete frequency axis in the Hilbert spectrum. Default: Same as the inverse of the sampling rate.
* `hht_amplitude_moving_avg_window` Window size used for the moving average filter applied to the instantaneous amplitude curves, measured in number of samples. Default: 5.
* `hht_frequency_moving_avg_window` Window size used for the moving average filter applied to the instantaneous frequency curves, measured in number of samples. Default: 41.
* `hht_split_signal_freq_change_enable` Enables splitting an IMF before Hilbert transforming if its dominating frequency changes abruptly if True. Default: True.
* `hht_split_signal_freq_change_threshold` Minimum amplitude of spike in gradient of dominating frequency for the IMF to be split at the spike before Hilbert transform. No effect if `hht_split_signal_freq_change_toggle` is False. Default: 0.5.
* `hht_split_signal_freq_change_nperseg` Window width of STFT measured in number of samples used for deciding where to split IMFs. No effect if `hht_split_signal_freq_change_toggle` is False. Default: 100.
* `minimum_frequency` Lower bound of frequency axis in Hilbert spectrum. Default: Inverse of the length of the segment plus extension padding (Nyquist-Shannon sampling theorem).
* `maximum_frequency` Upper bound of frequency axis in Hilbert spectrum. Default: No upper bound; the highest frequency found in the IMFs will be the highest value in the frequency axis.


### Hilbert spectrum analysis settings

* `minimum_total_non_zero_fraction` Minimum fraction of the combined row from the Hilbert spectrum that must be non-zero for the frequency band to be interpreted as a mode and analyzed. Default: 0.1.
* `minimum_consecutive_non_zero_length` Minimum consecutive series of non-zero elements in a combined row an element has to be part of to not be set to zero. Default:  5.
* `minimum_non_zero_improvement` Minimum improvement to number of non-zero elements in combined row from including another row for this new row to be included in the combined row. Default: 3.
* `max_coefficient_of_variation` Maximum coefficient of variation between an amplitude curve with interpolation and the fitted curve before the estimated damping is considered uncertain. Default: 0.4.
* `max_interp_fraction` Maximum fraction of the row elements that had to be filled with interpolation before the mode is considered uncertain. Default: 0.4.
* `skip_storing_uncertain_modes` Does not store details on modes with interpolation fraction exceeding max_interp_fraction if True. Stores the modes with an asterisk and brackets around the frequency in the CSV file if False. Default: False.
* `start_amp_curve_at_peak` Ignores all amplitude values before the maximum value in the row if this maximum is in the first half of the row’s non-zero portion if True. Default: True.


### Alarm system settings

* `alarm_median_amplitude_threshold` The minimum median amplitude of the mode in the segment for an alarm to be raised. Default: 0.
* `damping_ratio_weak_alarm_threshold` Damping ratio threshold for a weak alarm to be raised. Default: 0.15.
* `damping_ratio_strong_alarm_threshold` Damping ratio threshold for a strong alarm to be raised. Default: 0.05.
* `oscillation_timeout` Maximum number of seconds before end of segment the detected oscillation can end before it is interpreted as ended, and any alarm is suppressed. Default: 2.0.
* `segment_memory_freq_threshold` The tolerance range for frequency in both positive and negative directions for two modes in two consecutive segments to be considered the same mode. Used to determine if the mode is sustained. Default: 0.15.


### Result format settings

* `store_csv` Enables storing results table to CSV file if True. Default: True.
* `store_pkl` Enables storing results objects to PKL file if True. True.
* `csv_decimals` Number of decimals used in results table in PKL file. Default: 5.
* `csv_delimiter` Delimiter used between values in CSV file. Must be set to ";" instead of "," to open in Excel if Excel interprets "," as decimal separator (common in most European countries and others). Default: ",".
* `unit` Unit to be placed in "[]" after amplitudes in header of CSV table. Default: no unit or "[]" will be included.
* `include_advanced_results` Includes columns for advanced results in CSV if True. Only includes basic results if False. Default: False.



### Print settings

* `print_segment_number` Prints the segment number that is currently being analyzed to the console if True. Default: True.
* `print_emd_time` Prints the runtime of the EMD step of the segment analysis algorithm to the output console for each segment if True. Default: False.
* `print_hht_time` Prints the runtime of the whole HHT step of the segment analysis algorithm to the output console, including EMD, for each segment if True. Default: False.
* `print_segment_analysis_time` Prints the runtime of the whole segment analysis algorithm to the output console for each segment if True. Default: True.
* `print_alarms` Prints notice about alarms including their frequency to the output console when they occur. Default: True.
* `print_emd_sifting_details` Prints the number of sifting iterations for each IMF, along with REC for each of them, to the output console if True. Default: False.