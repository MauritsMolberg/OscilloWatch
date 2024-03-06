import pickle

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
        Constructor for the SignalSnapshotAnalysis class. Initializes variables and splits signal into segments.

        :param list | numpy.ndarray input_signal: Input signal that is to be split into segments and analyzed.
        :param AnalysisSettings settings: Object containing the settings for the different algorithms used in the signal
         analysis.
        """
        self.settings = settings
        self.input_signal = input_signal

        self.segment_list = []
        self.split_signal()

        self.segment_analysis_list = []

        self.results_path_updated = self.settings.results_file_path
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
        """
        Runs segment analysis on all the segments and stores the SegmentAnalysis objects in a list.
        :return: None
        """
        previous_segment = None
        for i, segment in enumerate(self.segment_list):
            if self.settings.print_segment_number:
                print(f"-------------------------------\nSegment {i}:")
            seg_analysis = SegmentAnalysis(segment, self.settings, previous_segment=previous_segment)
            seg_analysis.damping_analysis()

            seg_analysis.previous_segment = None  # To save storage space when storing in PKL file
            previous_segment = seg_analysis

            self.segment_analysis_list.append(seg_analysis)

    def write_results_to_csv(self):
        """
        Writes the estimated characteristics for all modes in all segments to CSV file.
        :return: None
        """
        headers = list(self.settings.blank_mode_info_dict)

        # Adds "_(number)" to file name if permission denied (when file is open in Excel, most likely)
        try:
            with open(self.results_path_updated + ".csv", 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=self.settings.csv_delimiter)

                csv_writer.writerow(["Segment"] + headers)
                for i, segment in enumerate(self.segment_analysis_list):
                    first_mode_in_segment = True
                    for data_dict in segment.mode_info_list:
                        # Make sure each segment number appears only once, for better readability
                        if first_mode_in_segment:
                            row = [i]
                            first_mode_in_segment = False
                        else:
                            row = [""]

                        for header in headers:
                            if isinstance(data_dict[header], float) or isinstance(data_dict[header], np.float64):
                                row.append(f"{data_dict[header]:.{self.settings.csv_decimals}f}")
                            else:
                                row.append(data_dict[header])
                        csv_writer.writerow(row)
                print(f"Results successfully saved to {self.results_path_updated}.csv.")
        except PermissionError:
            self.file_save_attempt_count += 1
            if self.file_save_attempt_count > 20:
                print("Unable to store results to csv.")
                return
            new_path = self.settings.results_file_path + "_" + str(self.file_save_attempt_count)
            print(f"Permission denied for file {self.results_path_updated}.csv. Trying to save to {new_path}.csv instead.")
            self.results_path_updated = new_path

            return self.write_results_to_csv()

    def write_result_objects_to_pkl(self):
        """
        Writes all SegmentAnalysis objects to PKL file.
        :return: None
        """
        try:
            with open(self.results_path_updated + ".pkl", "wb") as file:
                for segment_analysis_obj in self.segment_analysis_list:
                    pickle.dump(segment_analysis_obj, file)
                print(f"Result objects successfully written to {self.results_path_updated}.pkl.")
        except Exception as e:
            print(f"Results were not saved to {self.results_path_updated}.pkl, as the following exception "
                  f"occurred when writing results to pkl file: {e}.")
