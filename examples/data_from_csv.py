import matplotlib.pyplot as plt
import numpy as np

from OscilloWatch.SignalSnapshotAnalysis import SignalSnapshotAnalysis
from OscilloWatch.OWSettings import OWSettings
from OscilloWatch.csv_column_to_list import csv_column_to_list


file_path = "../example_pmu_data/Generator Loss event N45.CSV"
column_index = 1
fs = 50

data = csv_column_to_list(file_path, column_index, delimiter=";")
t = np.linspace(0, len(data)/fs, len(data))
plt.figure()
plt.plot(t, data)
plt.xlabel("Time [s]")
#plt.ylabel("Frequency [Hz]")
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
                            minimum_amplitude=0.001,
                            minimum_frequency=0.1,
                            results_file_path="../results/N45/gen/test"
                            )

snap_an = SignalSnapshotAnalysis(data, settings)
snap_an.analyze_whole_signal()

# for segment in sig_an.segment_analysis_list[26:31]:
#     segment.hht.emd.plot_emd_results(show=False)
#     segment.hht.plot_hilbert_spectrum(show=False)
#
plt.show()
