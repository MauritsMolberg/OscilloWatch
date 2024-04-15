import matplotlib.pyplot as plt
import numpy as np

from methods.SignalSnapshotAnalysis import SignalSnapshotAnalysis
from methods.AnalysisSettings import AnalysisSettings
from methods.csv_column_to_list import csv_column_to_list


# Analyze PMU data from a CSV file
if __name__ == "__main__":

    file_path = "../example_pmu_data/Real PMU signals 5min.CSV"
    column_index = 1
    fs = 50

    data = csv_column_to_list(file_path, column_index, delimiter=";")
    t = np.linspace(0, len(data)/fs, len(data))
    plt.figure()
    plt.plot(t, data)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.plot(t, data)
    #plt.show()

    settings = AnalysisSettings(
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
                                hht_amplitude_threshold=0.05,
                                results_file_path="../results/Real PMU data NTNU/NTNU_V"
                                )

    snap_an = SignalSnapshotAnalysis(data, settings)
    snap_an.analyze_whole_signal()
    snap_an.write_results_to_csv()
    snap_an.write_result_objects_to_pkl()

    # for segment in sig_an.segment_analysis_list[26:31]:
    #     segment.hht.emd.plot_emd_results(show=False)
    #     segment.hht.plot_hilbert_spectrum(show=False)
    #
    plt.show()
