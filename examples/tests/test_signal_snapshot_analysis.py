import numpy as np
import matplotlib.pyplot as plt
from OscilloWatch.OWSettings import OWSettings
from OscilloWatch.EMD import EMD
from OscilloWatch.HHT import HHT
from OscilloWatch.SignalSnapshotAnalysis import SignalSnapshotAnalysis

if __name__ == "__main__":
    np.random.seed(0)

    def f(t):
        return (
                20*np.exp(-.2*t)*np.cos(10*np.pi*t)
                + 8*np.exp(.1*t)*np.cos(5*np.pi*t)
                + 10*np.exp(.15*t)*np.cos(2.4*np.pi*t)
                + 16*np.exp(.1)*np.cos(np.pi*t)
                )

    def g(t):
        return 4*np.exp(.1*t)*np.cos(np.pi*t) + 15*np.exp(-.1*t)*np.cos(.4*np.pi*t)


    start = -20
    end = 22
    fs = 50

    t = np.arange(start, end, 1/fs)
    input_signal = f(t)

    #input_signal = np.load("k2a_with_controls_1s_fault_V.npy")
    #t = np.arange(0, len(input_signal)/fs, 1/fs)

    # plt.figure()
    # plt.plot(t, input_signal)
    # plt.axvline(x=0, ymin=0.1, ymax=.8, ls="--", color="red")
    # plt.axvline(x=10, ymin=0.1, ymax=.8, ls="--", color="red")
    # plt.xlabel("t [s]")
    # plt.tight_layout()
    # plt.show()

    settings = OWSettings(
                                fs=fs,
                                segment_length_time=10,
                                extension_padding_time_start=10,
                                extension_padding_time_end=2,
                                max_imfs=5,
                                print_emd_time=True,
                                print_hht_time=True,
                                include_asterisk_explanations=True,
                                include_advanced_results=True,
                                minimum_frequency=0.1
                                )

    sig_an = SignalSnapshotAnalysis(input_signal, settings)
    sig_an.analyze_whole_signal()

    for segment in sig_an.segment_analysis_list:
        segment.hht.emd.plot_emd_results(show=False)
        segment.hht.plot_hilbert_spectrum(show=False)

    #imf1 = sig_an.segment_analysis_list[1].hht.emd.imf_list[0]

    #plt.figure()
    #plt.plot(imf1)

    #hht1 = HHT(imf1, settings)
    #hht1.calc_hilbert_spectrum(imf1)
    #hht1.plot_hilbert_spectrum()

    plt.show()
