"""
This script is used to access the details from analysis by reading the .pkl file it has produced.
By default, it fetches the results from segment 17 and 18 after having run the script "data_from_csv_V.py", and plots
the EMD results, Hilbert spectra and mode details.
"""

import matplotlib.pyplot as plt
import numpy as np

from OscilloWatch.post_processing import read_from_pkl




if __name__ == "__main__":
    seg_res_list = read_from_pkl(file_path="../results/N45_b3000_V.pkl", start_index=17, end_index=18)

    for segment in seg_res_list:
        #plt.figure()
        #plt.plot(segment.input_signal)
        segment.hht.emd.plot_emd_results(show=False)
        segment.hht.plot_hilbert_spectrum(show=False)
        print(segment.mode_info_list)

    plt.show()
    