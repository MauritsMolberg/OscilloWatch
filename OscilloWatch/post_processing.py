import pickle

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
            if abs(mode["Frequency"] - mode_frequency) <= tolerance:
                found_mode = True
                amplitude_curve = np.append(amplitude_curve,
                                            [mode["Median amp."] for i in range(segment_length_samples)])
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
