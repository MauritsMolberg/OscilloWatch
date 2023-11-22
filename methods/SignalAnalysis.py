import numpy as np
import matplotlib.pyplot as plt
import csv
from methods.AnalysisSettings import AnalysisSettings
from methods.EMD import EMD
from methods.HHT import HHT
from methods.SegmentAnalysis import SegmentAnalysis


class SignalAnalysis:
    """
    Class that splits an input signal into segments and performs damping analysis on each segment. Simulates how the
    analysis would be performed on a real-time data stream.
    """
    def __init__(self, input_signal, settings: AnalysisSettings, results_file_path="results.csv"):
        self.settings = settings
        self.input_signal = input_signal

        self.segment_list = []
        self.split_signal()

        self.segment_analysis_list = []

        self.results_file_path = results_file_path

    def split_signal(self):
        """
        Splits signal into segments, and stores in object's segment_list variable. Includes the extra padding specified
        in settings, so there may be overlap.

        :return: None
        """
        for seg_ind in range((len(self.input_signal)
                             - self.settings.extra_padding_samples_start - self.settings.extra_padding_samples_end)
                             // self.settings.segment_length_samples):
            start_ind = seg_ind * self.settings.segment_length_samples
            end_ind = (seg_ind+1) * self.settings.segment_length_samples\
                      + self.settings.extra_padding_samples_start + self.settings.extra_padding_samples_end
            self.segment_list.append(self.input_signal[start_ind:end_ind])

    def analyze_whole_signal(self):
        for segment in self.segment_list:
            seg_analysis = SegmentAnalysis(segment, self.settings)
            seg_analysis.damping_analysis()
            self.segment_analysis_list.append(seg_analysis)

    def write_results_to_file(self):
        # Extracting headers from the first dictionary in the list
        headers = list(self.segment_analysis_list[0].damping_info_list[0].keys())

        with open(self.results_file_path, 'w', newline='') as csv_file:
            # Creating a CSV writer object with semicolon as the delimiter
            csv_writer = csv.writer(csv_file, delimiter=';')

            # Writing the headers to the CSV file
            csv_writer.writerow(headers)

            # Writing the values for each dictionary in the list
            for data_dict in list_of_dicts:
                # Writing each value separately
                csv_writer.writerow([data_dict[header] for header in headers])


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


    start = -1
    end = 51
    fs = 50
    t = np.arange(start, end, 1/fs)
    input_signal = f(t)
    #input_signal1 = np.load("k2a_with_controls.npy")

    settings = AnalysisSettings(segment_length_time=10, extra_padding_time_start=1, extra_padding_time_end=1)

    sig_an = SignalAnalysis(input_signal, settings)
    sig_an.analyze_whole_signal()

    for segment in sig_an.segment_analysis_list:
        segment.hht.plot_hilbert_spectrum(show=False)
        segment.hht.emd.plot_emd_results(show=False)


    plt.show()