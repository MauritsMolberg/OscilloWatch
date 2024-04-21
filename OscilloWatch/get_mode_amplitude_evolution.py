import numpy as np
import matplotlib.pyplot as plt

from OscilloWatch.SegmentAnalysis import SegmentAnalysis
from OscilloWatch.read_from_pkl import read_from_pkl
from OscilloWatch.csv_column_to_list import csv_column_to_list


def get_mode_amplitude_evolution(segment_list: list[SegmentAnalysis], mode_frequency, tolerance=None, fs=None,
                                 include_extension_start=True):
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
                interpolated_values = np.linspace(mode["Init. amp."], mode["Final amp."],
                                                  num=int((mode["End time"]-mode["Start time"])*fs))
                amplitude_curve = np.append(amplitude_curve,
                                            np.concatenate((np.zeros(int(mode["Start time"]*fs)),
                                                            interpolated_values,
                                                            np.zeros(int((segment_length_time - mode["End time"])*fs))))
                                            )
                break
        if not found_mode:
            amplitude_curve = np.append(amplitude_curve, np.zeros(segment_length_samples))
    # Fill last few samples with zeros
    missing = extension_start + segment_length_samples*len(segment_list) - len(amplitude_curve)
    amplitude_curve = np.concatenate((amplitude_curve, np.zeros(missing)))
    return amplitude_curve
