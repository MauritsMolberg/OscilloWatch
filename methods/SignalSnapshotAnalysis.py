import numpy as np
import csv
import os
from methods.AnalysisSettings import AnalysisSettings
from methods.SegmentAnalysis import SegmentAnalysis


class SignalSnapshotAnalysis:
    """
    Class that splits an input signal into segments and performs damping analysis on each segment. Simulates how the
    analysis would be performed on a real-time data stream.
    """
    def __init__(self, input_signal, settings: AnalysisSettings):
        """
        Constructor for the SignalSnapshotAnalysis class.

        :param input_signal:
        :type input_signal: list or numpy.ndarray
        :param AnalysisSettings settings:
        """
        self.settings = settings
        self.input_signal = input_signal

        self.segment_list = []
        self.split_signal()

        self.segment_analysis_list = []

        self.file_save_attempt_count = 0

    def split_signal(self):
        """
        Splits signal into segments, and stores in object's segment_list variable. Includes the extra padding specified
        in settings, so there may be overlap.

        :return: None
        """
        for seg_ind in range((len(self.input_signal) - self.settings.extension_padding_samples_start
                              - self.settings.extension_padding_samples_end)
                             // self.settings.segment_length_samples):
            start_ind = seg_ind * self.settings.segment_length_samples
            end_ind = ((seg_ind+1)*self.settings.segment_length_samples
                       + self.settings.extension_padding_samples_start + self.settings.extension_padding_samples_end)
            self.segment_list.append(self.input_signal[start_ind:end_ind])

        remaining_samples = len(self.input_signal) - end_ind
        print(f"Input signal split into {seg_ind+1} segments.")
        if remaining_samples:
            print(f"Last {remaining_samples/self.settings.fs + self.settings.extension_padding_time_end} seconds "
                  f"excluded.")
            if self.settings.extension_padding_time_end:
                print(f"({self.settings.extension_padding_time_end} of which were used as padding for last segment.)")

    def analyze_whole_signal(self):
        for i, segment in enumerate(self.segment_list):
            if self.settings.print_segment_number:
                print(f"-------------------------------\nSegment {i+1}:")
            seg_analysis = SegmentAnalysis(segment, self.settings)
            seg_analysis.damping_analysis()
            self.segment_analysis_list.append(seg_analysis)

    def write_results_to_csv(self, file_path="default"):
        headers = list(self.settings.blank_mode_info_dict)

        if file_path == "default":
            current_file_path = self.settings.results_file_path + ".csv"
        else:
            current_file_path = file_path

        # Adds "_(number)" to file name if permission denied (when file is open in Excel, most likely)
        try:
            with open(current_file_path, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=self.settings.csv_delimiter)

                csv_writer.writerow(["Segment"] + headers)
                for i, segment in enumerate(self.segment_analysis_list):
                    first_mode_in_segment = True
                    for data_dict in segment.mode_info_list:
                        # Make sure each segment number appears only once, for better readability
                        if first_mode_in_segment:
                            row = [i + 1]
                            first_mode_in_segment = False
                        else:
                            row = [""]

                        for header in headers:
                            if isinstance(data_dict[header], float) or isinstance(data_dict[header], np.float64):
                                row.append(f"{data_dict[header]:.{self.settings.csv_decimals}f}")
                            else:
                                row.append(data_dict[header])
                        csv_writer.writerow(row)
        except PermissionError:
            self.file_save_attempt_count += 1
            if self.file_save_attempt_count > 20:
                print("Unable to store results to csv.")
                return
            new_path = self.settings.results_file_path + "_" + str(self.file_save_attempt_count) + ".csv"
            print(f"Permission denied for file {current_file_path}. Trying to save to {new_path} instead.")

            return self.write_results_to_csv(new_path)

        print(f"Results successfully saved to {current_file_path}.")
        return

# Todo: Incorporate frequency result changes
