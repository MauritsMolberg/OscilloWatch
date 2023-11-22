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


    settings = AnalysisSettings(remove_padding_after_emd=True, max_imfs=3, extra_padding_time_start=2, extra_padding_time_end=2)
    start = -1
    end = 11
    fs = 50
    t = np.arange(start, end, 1/fs)
    input_signal1 = f(t)
    #input_signal1 = np.load("k2a_with_controls.npy")


    emd1 = EMD(input_signal1, settings)
    emd1.perform_emd()
    emd1.plot_emd_results(show=False)

    settings.remove_padding_after_emd=False

    hht = HHT(input_signal1, settings)
    hht.full_hht()
    hht.plot_hilbert_spectrum(show=False)

    settings.print_emd_time = True
    settings.print_hht_time = True
    settings.print_segment_analysis_time = True

    damp = SegmentAnalysis(input_signal1, settings)
    damp.damping_analysis()

    for i in range(len(damp.damping_info_list)):
        print(damp.damping_info_list[i], "\n")

    plt.show()