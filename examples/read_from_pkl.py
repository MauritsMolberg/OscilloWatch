import pickle
import matplotlib.pyplot as plt
import numpy as np


def read_from_pkl(file_path="results.pkl"):
    try:
        with open(file_path, 'rb') as file:
            loaded_segment_results = []
            while True:
                loaded_segment = pickle.load(file)
                loaded_segment_results.append(loaded_segment)
    except EOFError:
        pass  # End of file reached

    return loaded_segment_results


if __name__ == "__main__":
    seg_res_list = read_from_pkl("../results/results_1.pkl")
    for segment in seg_res_list[:3]:
        plt.figure()
        plt.plot(segment.input_signal)
        segment.hht.emd.plot_emd_results(show=False)
        segment.hht.plot_hilbert_spectrum(show=False)

    plt.show()
