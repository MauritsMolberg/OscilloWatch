import numpy as np
import matplotlib.pyplot as plt
from methods.AnalysisSettings import AnalysisSettings
from methods.EMD import EMD
from methods.HHT import HHT
from methods.SignalSnapshotAnalysis import SignalSnapshotAnalysis

if __name__ == "__main__":
    np.random.seed(0)

    def f(t):
        #return 6*np.exp(.2*t)*np.cos(3*np.pi*t) + 15*np.exp(-.1*t)*np.cos(np.pi*t)
        return (
                10*np.exp(.15*t)*np.cos(2.4*np.pi*t)
                #+ 16*np.exp(.1)*np.cos(np.pi*t)
                + 8*np.exp(.1*t)*np.cos(5*np.pi*t)
                + 20*np.exp(-.2*t)*np.cos(10*np.pi*t)
                )

    def g(t):
        return 4*np.exp(.1*t)*np.cos(np.pi*t) + 15*np.exp(-.1*t)*np.cos(.4*np.pi*t)


    start = -10
    end = 180
    fs = 50

    t = np.arange(start, end, 1/fs)
    input_signal = f(t)

    #input_signal = np.load("k2a_with_controls_1s_fault_V.npy")
    #t = np.arange(0, len(input_signal)/fs, 1/fs)

    # fig, ax = plt.subplots()
    # ax.plot(t, input_signal)
    # ax.set_title("Input signal")
    # plt.tight_layout()

    settings = AnalysisSettings(
                                fs=fs,
                                segment_length_time=10,
                                extension_padding_time_start=10,
                                extension_padding_time_end=2,
                                mirror_padding_fraction=1,
                                start_amp_curve_at_peak=True,
                                print_emd_time=True,
                                print_hht_time=True,
                                print_emd_sifting_details=False,
                                print_segment_number=True,
                                print_segment_analysis_time=True,
                                hht_split_signal_freq_change_toggle=True,
                                max_imfs=4,
                                hht_frequency_moving_avg_window=41
                                )

    sig_an = SignalSnapshotAnalysis(input_signal, settings)
    sig_an.analyze_whole_signal()
    sig_an.write_results_to_csv()
    sig_an.write_result_objects_to_pkl()

    # for segment in sig_an.segment_analysis_list:
    #     segment.hht.emd.plot_emd_results(show=False)
    #     segment.hht.plot_hilbert_spectrum(show=False)

    #imf1 = sig_an.segment_analysis_list[1].hht.emd.imf_list[0]

    #plt.figure()
    #plt.plot(imf1)

    #hht1 = HHT(imf1, settings)
    #hht1.calc_hilbert_spectrum(imf1)
    #hht1.plot_hilbert_spectrum()

    plt.show()
