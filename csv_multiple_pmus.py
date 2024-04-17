from time import time

import matplotlib.pyplot as plt
import numpy as np

from OscilloWatch.SignalSnapshotAnalysis import SignalSnapshotAnalysis
from OscilloWatch.AnalysisSettings import AnalysisSettings
from OscilloWatch.csv_column_to_list import csv_column_to_list

if __name__ == "__main__":

    start_time = time()

    file_path = "example_pmu_data/180924-osc-frekvens og vinkel.csv"


    column_index = 1
    fs = 50

    data = csv_column_to_list(file_path, column_index)
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
                                hht_split_signal_freq_change_toggle=True,
                                max_imfs=5,
                                skip_storing_uncertain_modes=False,
                                hht_amplitude_threshold=0.001,
                                results_file_path="results/fortun"
                                )

    snap_an = SignalSnapshotAnalysis(data, settings)
    snap_an.analyze_whole_signal()
    snap_an.write_results_to_csv()
    snap_an.write_result_objects_to_pkl()


    column_index = 3
    fs = 10

    data = csv_column_to_list(file_path, column_index)
    t = np.linspace(0, len(data)/fs, len(data))
    plt.figure()
    plt.plot(t, data)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.plot(t, data)

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
        hht_split_signal_freq_change_toggle=True,
        max_imfs=5,
        skip_storing_uncertain_modes=False,
        hht_amplitude_threshold=0.001,
        results_file_path="results/hasle"
    )
    snap_an = SignalSnapshotAnalysis(data, settings)
    snap_an.analyze_whole_signal()
    snap_an.write_results_to_csv()
    snap_an.write_result_objects_to_pkl()


    column_index = 5

    data = csv_column_to_list(file_path, column_index)
    t = np.linspace(0, len(data)/fs, len(data))
    plt.figure()
    plt.plot(t, data)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.plot(t, data)

    settings.results_file_path = "results/klæbu"
    snap_an = SignalSnapshotAnalysis(data, settings)
    snap_an.analyze_whole_signal()
    snap_an.write_results_to_csv()
    snap_an.write_result_objects_to_pkl()


    column_index = 7

    data = csv_column_to_list(file_path, column_index)
    t = np.linspace(0, len(data)/fs, len(data))
    plt.figure()
    plt.plot(t, data)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.plot(t, data)

    settings.fs = fs
    settings.results_file_path = "results/kristiansand"
    snap_an = SignalSnapshotAnalysis(data, settings)
    snap_an.analyze_whole_signal()
    snap_an.write_results_to_csv()
    snap_an.write_result_objects_to_pkl()


    column_index = 9

    data = csv_column_to_list(file_path, column_index)
    t = np.linspace(0, len(data)/fs, len(data))
    plt.figure()
    plt.plot(t, data)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.plot(t, data)

    settings.fs = fs
    settings.results_file_path = "results/kvandal"
    snap_an = SignalSnapshotAnalysis(data, settings)
    snap_an.analyze_whole_signal()
    snap_an.write_results_to_csv()
    snap_an.write_result_objects_to_pkl()


    column_index = 11

    data = csv_column_to_list(file_path, column_index)
    t = np.linspace(0, len(data)/fs, len(data))
    plt.figure()
    plt.plot(t, data)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.plot(t, data)

    settings.fs = fs
    settings.results_file_path = "results/kvilldal"
    snap_an = SignalSnapshotAnalysis(data, settings)
    snap_an.analyze_whole_signal()
    snap_an.write_results_to_csv()
    snap_an.write_result_objects_to_pkl()


    column_index = 13
    fs = 50

    data = csv_column_to_list(file_path, column_index)
    t = np.linspace(0, len(data)/fs, len(data))
    plt.figure()
    plt.plot(t, data)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.plot(t, data)

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
        hht_split_signal_freq_change_toggle=True,
        max_imfs=5,
        skip_storing_uncertain_modes=False,
        hht_amplitude_threshold=0.001,
        results_file_path="results/skaidi"
    )
    snap_an = SignalSnapshotAnalysis(data, settings)
    snap_an.analyze_whole_signal()
    snap_an.write_results_to_csv()
    snap_an.write_result_objects_to_pkl()


    column_index = 14
    fs = 10

    data = csv_column_to_list(file_path, column_index)
    t = np.linspace(0, len(data)/fs, len(data))
    plt.figure()
    plt.plot(t, data)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.plot(t, data)

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
        hht_split_signal_freq_change_toggle=True,
        max_imfs=5,
        skip_storing_uncertain_modes=False,
        hht_amplitude_threshold=0.001,
        results_file_path="results/varangerbotn"
    )
    snap_an = SignalSnapshotAnalysis(data, settings)
    snap_an.analyze_whole_signal()
    snap_an.write_results_to_csv()
    snap_an.write_result_objects_to_pkl()

    print(f"Completed in {time()-start_time:.3f} seconds")

    plt.show()