import csv
import matplotlib.pyplot as plt
import numpy as np

from methods.SignalSnapshotAnalysis import SignalSnapshotAnalysis
from methods.AnalysisSettings import AnalysisSettings


def csv_column_to_list(file_path, column_index, delimiter=","):
    """
    Simple function for reading from a CSV file and putting all non-NaN elements in a list. The top row is assumed to
    contain headers and will not be included in the list.

    :param str file_path: File path to the CSV file that is to be read.
    :param int column_index: Index of the row that is to be read and put in a list.
    :param str delimiter: Symbol used as delimiter in the CSV file. Comma is the most common, but semicolon is typically
        used in countries where the comma is used for decimal numbers.
    :return: List containing non-NaN elements of the selected row from the CSV file.
    :rtype: list
    """
    values = []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        headers = next(csv_reader)  # Skip the header row
        for row in csv_reader:
            if row[column_index].lower() != "nan" and len(row) > column_index:
                values.append(float(row[column_index]))
    return values


if __name__ == "__main__":

    file_path = "../example_pmu_data/180924-osc-frekvens og vinkel.csv"
    column_index = 1
    fs = 50

    data = csv_column_to_list(file_path, column_index)
    t = np.linspace(0, len(data)/fs/60, len(data))
    # plt.figure()
    # plt.plot(t, data)
    # plt.xlabel("Time [min]")
    # plt.ylabel("Frequency [Hz]")
    # plt.plot(t, data)

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

    snap_an = SignalSnapshotAnalysis(data, settings)
    snap_an.analyze_whole_signal()
    snap_an.write_results_to_csv()
    snap_an.write_result_objects_to_pkl()

    # for segment in sig_an.segment_analysis_list[26:31]:
    #     segment.hht.emd.plot_emd_results(show=False)
    #     segment.hht.plot_hilbert_spectrum(show=False)
    #
    plt.show()
