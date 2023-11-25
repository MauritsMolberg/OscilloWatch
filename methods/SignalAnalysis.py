import numpy as np
import csv
import os
from methods.AnalysisSettings import AnalysisSettings
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
        self.file_save_attemt_count = 0

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
        headers = ["Warning",
                    "Freq. band start",
                    "Freq. band stop",
                    "Start time",
                    "End time",
                    "Non zero fraction",
                    "Initial amplitude",
                    "Final amplitude",
                    "Initial amplitude estimate",
                    "Decay rate",
                    "Damping ratio",
                    "Interpolated fraction",
                    "Coefficient of variation",
                    "Note"]

        # Adds _new to file name if permission denied (likely already open in Excel)
        try:
            with open(self.results_file_path, 'w', newline='') as csv_file:
                pass
        except PermissionError:
            if self.file_save_attemt_count > 9:
                print("Unable to save results to file.")
                return
            new_path = os.path.splitext(self.results_file_path)[0] +"_new"+ os.path.splitext(self.results_file_path)[1]
            print(f"Permission denied for file {self.results_file_path}. Trying to save to {new_path} instead.")
            self.results_file_path = new_path
            self.file_save_attemt_count += 1
            return self.write_results_to_file()

        with open(self.results_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=';')

            csv_writer.writerow(["Segment"] + headers)
            for i, segment in enumerate(self.segment_analysis_list):
                for data_dict in segment.oscillation_info_list:
                    row = [i+1]
                    for header in headers:
                        if type(data_dict[header]) == float or type(data_dict[header]) == np.float64:
                            row.append(f"{data_dict[header]:.{self.settings.csv_decimals}f}")
                        else:
                            row.append(data_dict[header])
                    csv_writer.writerow(row)
        print(f"Results successfully saved to {self.results_file_path}.")
        return
