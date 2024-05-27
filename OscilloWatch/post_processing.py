import pickle
import csv

import numpy as np

from OscilloWatch.SegmentAnalysis import SegmentAnalysis


def read_from_pkl(file_path="../results/results.pkl", start_index=0, end_index=None):
    """
    Read analysis results from PKL file and return a list of the SegmentAnalysis objects.

    :param str file_path: File path to the PKL file to read from.
    :param int start_index: Index of the first segment to include in the returned list.
    :param int | None end_index: Index of the last segment to include in the returned list. Default: Last segment.
    :return: List of SegmentAnalysis objects from the PKL file.
    :rtype: list[SegmentAnalysis]
    """
    try:
        with open(file_path, 'rb') as file:
            loaded_objects = []
            current_index = 0

            while True:
                try:
                    loaded_object = pickle.load(file)
                except EOFError:
                    break  # End of file reached

                if current_index >= start_index and (end_index is None or current_index <= end_index):
                    loaded_objects.append(loaded_object)

                current_index += 1

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error occurred while reading from '{file_path}': {e}")
        return []

    return loaded_objects


def get_mode_amplitude_evolution(segment_list: list[SegmentAnalysis],
                                 mode_frequency,
                                 tolerance=None,
                                 fs=None,
                                 include_extension_start=True):
    """
    Get an array with the temporal evolution of the median amplitude per segment of a mode.

    :param list[SegmentAnalysis] segment_list: List of SegmentAnalysis objects to extract the amplitude values from.
    :param float mode_frequency: Approximate frequency in Hz of the mode to analyze.
    :param float tolerance: Tolerance of the frequency of a mode to be considered the same mode as the specified
     mode_frequency. Default: Same as in the settings of the SegmentAnalysis objects.
    :param int fs: Sampling rate of the signal. Default: same as in the settings of the SegmentAnalysis objects.
    :param bool include_extension_start: Includes extension padding at the start of the signal if True. Starts directly
     at the start of the first segment if False.
    :return: Temporal evolution of the median amplitude of the mode.
    :rtype: numpy.ndarray[numpy.float64]
    """
    if tolerance is None:
        tolerance = segment_list[0].settings.segment_memory_freq_threshold
    if fs is None:
        fs = segment_list[0].settings.fs
    segment_length_time = segment_list[0].settings.segment_length_time
    segment_length_samples = segment_length_time*fs

    if include_extension_start:
        extension_start = segment_list[0].settings.extension_padding_time_start*fs
    else:
        extension_start = 0

    amplitude_curve = np.zeros(extension_start)

    for segment in segment_list:
        found_mode = False

        for mode in segment.mode_info_list:
            if abs(mode["frequency"] - mode_frequency) <= tolerance:
                found_mode = True
                amplitude_curve = np.append(amplitude_curve,
                                            [mode["median_amp"] for i in range(segment_length_samples)])
                break
        if not found_mode:
            amplitude_curve = np.append(amplitude_curve, np.zeros(segment_length_samples))
    # Fill last few samples with zeros
    missing = extension_start + segment_length_samples*len(segment_list) - len(amplitude_curve)
    amplitude_curve = np.concatenate((amplitude_curve, np.zeros(missing)))
    return amplitude_curve


def reconstruct_signal(seg_res_list):
    """
    Reconstruct the original time-series signal used to analyze a set of segments. Does not include padding from the
    beginning and end of the signal.

    :param list[SegmentAnalysis] seg_res_list: List of SegmentAnalysis objects whose signals should be included in the
     reconstructed signal.
    :return: Reconstructed signal.
    :rtype: numpy.ndarray[numpy.float64]
    """
    signal = np.array([])
    settings = seg_res_list[0].settings
    for segment in seg_res_list:
        segment_signal = segment.input_signal[settings.extension_padding_samples_start:
                                              -settings.extension_padding_samples_end]
        signal = np.append(signal, segment_signal)
    return signal


def summarize_alarms(pkl_file_tuple_list,
                     results_file_path="alarms_summary.csv",
                     include_timestamp=True,
                     csv_delimiter=","):
    """
    Stores info on all alarms that were raised from analysis of data from multiple PMUs analyzed with the same settings
    into a csv file.

    :param list[tuple[str, str]] pkl_file_tuple_list: List of tuples containing the file path to a PKL file and the name
     of the corresponding PMU. Must be on the form [('file_path_1', 'pmu_1_name'), ('file_path_2', 'pmu_2_name') ...].
     The PKL files must contain the same number of segments.
    :param str results_file_path: File path to CSV file to store results to, including the .csv extension.
    :param str csv_delimiter: Delimiter to use for separating columns in CSV file.
    :return: None
    :rtype: NoneType
    """
    pkl_file_tuple_list_unpacked = list(zip(*pkl_file_tuple_list))

    # Read PKL files
    seg_res_list_list = []
    for pkl_file in pkl_file_tuple_list_unpacked[0]:
        seg_res_list_list.append(read_from_pkl(pkl_file))
    settings = seg_res_list_list[0][0].settings

    if settings.unit is None:
        unit_string = ""
    else:
        unit_string = f"[{settings.unit}]"

    row_list = [["Segment", "PMU", "Frequency [Hz]", "Median amplitude" + unit_string, "Alarm type"]]  # Headers for CSV

    # Raise error if not equal number of segments from the PMUs
    length = len(seg_res_list_list[0])
    for sublist in seg_res_list_list:
        if len(sublist) != length:
            raise ValueError("Number of segments from each PMU is not equal")

    # Transpose seg_res_list_list, to read segment data in the correct order
    seg_res_list_list_transposed = []
    for i in range(length):
        seg_res_list_list_transposed.append([seg_res_list_list[j][i] for j in range(len(seg_res_list_list))])

    # Iterate through segments for each PMU
    for segment_idx, segment_list in enumerate(seg_res_list_list_transposed):  # For each segment
        alarm_flag = False
        for pmu_idx, segment_data in enumerate(segment_list):  # For each PMU's data on that segment
            first_mode_in_segment = True
            for mode in segment_data.mode_info_list:
                if mode["alarm"] and mode["alarm"] != "No alarm, ended early":
                    if first_mode_in_segment:
                        row_new = ["",
                                   pkl_file_tuple_list_unpacked[1][pmu_idx],
                                   f"{mode['frequency']:.2f}",
                                   f"{mode['median_amp']:.5f}",
                                   mode["alarm"]]
                        first_mode_in_segment = False
                    else:
                        row_new = ["",
                                   "",
                                   f"{mode['frequency']:.2f}",
                                   f"{mode['median_amp']:.5f}",
                                   mode["alarm"]]
                    if not alarm_flag:
                        if include_timestamp:
                            timestamp = (settings.extension_padding_time_start
                                         + segment_idx * settings.segment_length_time)
                            row_new[0] = f"{segment_idx} ({timestamp} s)"
                        else:
                            row_new[0] = segment_idx
                        alarm_flag = True
                    row_list.append(row_new)

    # Write results to CSV file
    with open(results_file_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=csv_delimiter)
        for row in row_list:
            csv_writer.writerow(row)
        print(f"Alarm summary written to {results_file_path}.")
