import numpy as np
import matplotlib.pyplot as plt

from methods.SegmentAnalysis import SegmentAnalysis
from methods.read_from_pkl import read_from_pkl
from methods.csv_column_to_list import csv_column_to_list


def get_mode_amplitude_evolution(segment_list: list[SegmentAnalysis], mode_frequency, tolerance=None):
    if tolerance is None:
        tolerance = segment_list[0].settings.segment_memory_freq_threshold
    fs = segment_list[0].settings.fs
    segment_length_samples = segment_list[0].settings.segment_length_samples
    segment_length_time = segment_list[0].settings.segment_length_time

    amplitude_curve = np.zeros(segment_list[0].settings.extension_padding_samples_start)

    for segment in segment_list:
        found_mode = False
        for mode in segment.mode_info_list:
            if abs(mode["Frequency"] - mode_frequency) <= tolerance:
                interpolated_values = np.linspace(mode["Init. amp."], mode["Final amp."],
                                                  num=int((mode["End time"]-mode["Start time"])*fs))
                amplitude_curve = np.append(amplitude_curve,
                                            np.concatenate((np.zeros(int(mode["Start time"]*fs)),
                                                            interpolated_values,
                                                            np.zeros(int((segment_length_time - mode["End time"])*fs))))
                                            )
                # plt.plot(amplitude_curve)
                # plt.show()
                found_mode = True
                break
        if not found_mode:
            amplitude_curve = np.append(amplitude_curve, np.zeros(segment_length_samples))
    return amplitude_curve


if __name__ == "__main__":
    mode_freq = 0.4

    seg_lst = read_from_pkl("../results/.utfall P NO-SE.pkl")
    amplitude_curve = get_mode_amplitude_evolution(seg_lst, mode_freq)

    tAxis_amp = np.linspace(0,
                            (seg_lst[0].settings.extension_padding_time_start
                         + seg_lst[0].settings.segment_length_time * len(seg_lst)),
                            len(amplitude_curve))
    full_signal = csv_column_to_list("../example_pmu_data/utfall Olkiluoto.csv", 6, delimiter=";")
    tAxis = np.arange(0, len(full_signal)/seg_lst[0].settings.fs, 1/seg_lst[0].settings.fs)

    plt.figure()
    plt.title("Time series plot")
    plt.plot(tAxis, full_signal)
    plt.ylabel("Active power")
    plt.xlabel("Time [s]")

    plt.figure()
    plt.title(f"Amplitude of {mode_freq} Hz mode")
    plt.plot(tAxis_amp, amplitude_curve)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()