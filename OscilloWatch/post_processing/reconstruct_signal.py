import numpy as np
from OscilloWatch.SegmentAnalysis import SegmentAnalysis


def reconstruct_signal(seg_res_list: list[SegmentAnalysis]):
    signal = np.array([])
    settings = seg_res_list[0].settings
    for segment in seg_res_list:
        segment_signal = segment.input_signal[settings.extension_padding_samples_start:
                                              -settings.extension_padding_samples_end]
        signal = np.append(signal, segment_signal)
    return signal


if __name__ == "__main__":
    from OscilloWatch.post_processing.read_from_pkl import read_from_pkl
    import matplotlib.pyplot as plt

    seg_res_list = read_from_pkl("../../results/Real-time/real_NTNU_f.pkl")
    signal = reconstruct_signal(seg_res_list)

    fs = seg_res_list[0].settings.fs
    t = np.linspace(0, len(signal)/fs, len(signal))

    plt.figure()
    plt.plot(t, signal)
    plt.show()
