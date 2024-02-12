import csv
import matplotlib.pyplot as plt
import numpy as np

from methods.SignalSnapshotAnalysis import SignalSnapshotAnalysis
from methods.AnalysisSettings import AnalysisSettings


def read_csv_column(file_path, column_index):
    values = []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        headers = next(csv_reader)  # Skip the header row
        for row in csv_reader:
            if row[column_index].lower() != "nan" and len(row) > column_index:
                values.append(float(row[column_index]))
    return values


file_path = "180924-osc-frekvens og vinkel.csv"
column_index = 1

data = read_csv_column(file_path, column_index)
t = np.linspace(0, len(data)/50/60, len(data))
plt.figure()
plt.plot(t, data)
plt.xlabel("Time [min]")
#plt.ylabel("Frequency [Hz]")
plt.plot(t, data)

settings = AnalysisSettings(
                            fs=50,
                            segment_length_time=10,
                            extension_padding_time_start=10,
                            extension_padding_time_end=2,
                            print_segment_analysis_time=True,
                            start_amp_curve_at_peak=True,
                            print_segment_number=True,
                            print_emd_sifting_details=False,
                            hht_frequency_moving_avg_window=41,
                            hht_split_signal_freq_change_toggle=True,
                            max_imfs=5,
                            skip_storing_uncertain_results=False,
                            hht_amplitude_threshold=0.001
                            )

sig_an = SignalSnapshotAnalysis(data, settings)
sig_an.analyze_whole_signal()
sig_an.write_results_to_file()

for segment in sig_an.segment_analysis_list[26:31]:
    segment.hht.emd.plot_emd_results(show=False)
    segment.hht.plot_hilbert_spectrum(show=False)

plt.show()
