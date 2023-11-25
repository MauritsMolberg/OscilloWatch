import numpy as np
import matplotlib.pyplot as plt
from methods.SegmentAnalysis import SegmentAnalysis
from methods.HHT import HHT
from methods.EMD import EMD
from methods.AnalysisSettings import AnalysisSettings

if __name__ == "__main__":
    np.random.seed(0)
    def f(t):
        #return 6*np.exp(.2*t)*np.cos(3*np.pi*t) + 15*np.exp(-.1*t)*np.cos(np.pi*t)
        return (10*np.exp(.2*t)*np.cos(2.4*np.pi*t)
                #+ 8*np.exp(-.1*t)*np.cos(np.pi*t)
                #+ 3*np.exp(.3*t)*np.cos(5*np.pi*t)
                + 20*np.exp(-.2*t)*np.cos(10*np.pi*t))


    def g(t):
        return 4*np.exp(.2*t)*np.cos(3*np.pi*t)


    settings = AnalysisSettings(max_imfs=3, extra_padding_time_start=2, extra_padding_time_end=2)
    start = -1
    end = 11
    fs = 50
    t = np.arange(start, end, 1/fs)
    input_signal1 = f(t)
    #input_signal1 = np.load("k2a_with_controls.npy")


    settings.print_emd_time = True
    settings.print_hht_time = True
    settings.print_segment_analysis_time = True

    damp = SegmentAnalysis(input_signal1, settings)
    damp.damping_analysis()
    damp.hht.emd.plot_emd_results(show=False, include_padding=True)
    damp.hht.plot_hilbert_spectrum(show=False)

    for i in range(len(damp.oscillation_info_list)):
        print(damp.oscillation_info_list[i], "\n")

    plt.show()