import matplotlib.pyplot as plt
import numpy as np

from OscilloWatch.post_processing import read_from_pkl


if __name__ == "__main__":
    seg_res_list = read_from_pkl("../../results/N45/gen/b5110_angle.pkl.pkl", 16, 17)

    for segment in seg_res_list:
        #plt.figure()
        #plt.plot(segment.input_signal)
        segment.hht.emd.plot_emd_results(show=False)
        segment.hht.plot_hilbert_spectrum(show=False)
        print(segment.mode_info_list)

    plt.show()
    