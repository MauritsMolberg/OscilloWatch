import numpy as np
import matplotlib.pyplot as plt
from methods.AnalysisSettings import AnalysisSettings
from methods.EMD import EMD
from methods.HHT import HHT
from methods.SignalAnalysis import SignalAnalysis

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


    start = -12
    end = 51
    fs = 50

    t = np.arange(start, end, 1/fs)
    input_signal = f(t)

    #input_signal = np.load("k2a_with_controls_1s_fault_V.npy")
    #t = np.arange(0, len(input_signal))

    fig, ax = plt.subplots()
    ax.plot(t, input_signal)
    ax.set_title("Input signal")

    settings = AnalysisSettings(segment_length_time=10,
                                extra_padding_time_start=2,
                                extra_padding_time_end=1,
                                start_amp_curve_at_peak=True)

    sig_an = SignalAnalysis(input_signal, settings)
    sig_an.analyze_whole_signal()
    sig_an.write_results_to_file()

    for segment in sig_an.segment_analysis_list:
        segment.hht.emd.plot_emd_results(show=False)
        segment.hht.plot_hilbert_spectrum(show=False)



    plt.show()