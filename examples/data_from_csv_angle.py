import matplotlib.pyplot as plt
import numpy as np

from OscilloWatch.SignalSnapshotAnalysis import SignalSnapshotAnalysis
from OscilloWatch.OWSettings import OWSettings
from OscilloWatch.csv_column_to_list import csv_column_to_list


file_path = "../example_pmu_data/Generator Loss event N45.CSV"

reference_angle = np.unwrap(np.array(csv_column_to_list(file_path, 2, delimiter=";")), period=360)



column_index = 30
fs = 50

angle_data = reference_angle - np.unwrap(np.array(csv_column_to_list(file_path,column_index,delimiter=";")),period=360)
t = np.linspace(0, len(angle_data)/fs, len(angle_data))
plt.figure()
plt.plot(t, angle_data)
plt.xlabel("Time [s]")
plt.ylabel("Angle [deg]")
#plt.show()

settings = OWSettings(
                            fs=fs,
                            segment_length_time=10,
                            extension_padding_time_start=10,
                            extension_padding_time_end=2,
                            print_segment_analysis_time=True,
                            start_amp_curve_at_peak=True,
                            print_segment_number=True,
                            print_emd_sifting_details=False,
                            hht_frequency_moving_avg_window=41,
                            max_imfs=5,
                            skip_storing_uncertain_modes=False,
                            minimum_amplitude=0.01,
                            alarm_median_amplitude_threshold=1.0,
                            minimum_frequency=0.1,
                            results_file_path="../results/N45/gen/b5110_angle"
                            )

snap_an = SignalSnapshotAnalysis(angle_data, settings)
snap_an.analyze_whole_signal()

plt.show()
