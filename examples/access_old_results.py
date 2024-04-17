import matplotlib.pyplot as plt
import numpy as np

from OscilloWatch.read_from_pkl import read_from_pkl


if __name__ == "__main__":
    seg_res_list = read_from_pkl("../results/Real PMU data NTNU/NTNU_V.pkl", 197, 198)

    for segment in seg_res_list:
        #plt.figure()
        #plt.plot(segment.input_signal)
        segment.hht.emd.plot_emd_results(show=False)
        segment.hht.plot_hilbert_spectrum(show=False)

    plt.show()
    