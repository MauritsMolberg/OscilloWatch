#################################################################################
#     OscilloWatch, early warning application for low-frequency oscillations
#     Copyright (C) 2024  Maurits Sørensen Molberg
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
#################################################################################

import pickle
import os
import csv

import numpy as np

from OscilloWatch.OWSettings import OWSettings
from OscilloWatch.SegmentAnalysis import SegmentAnalysis


class SignalSnapshotAnalysis:
    """
    Class that splits an input signal into segments and performs damping analysis on each segment. Simulates how the
    analysis would be performed on a real-time data stream.
    """

    def __init__(self, input_signal, settings: OWSettings):
        """
        Constructor for the SignalSnapshotAnalysis class. Initializes variables and splits signal into segments.

        :param list | numpy.ndarray input_signal: Input signal that is to be split into segments and analyzed.
        :param OWSettings settings: Object containing the settings for the different algorithms used in the signal
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
        if (len(self.input_signal) <
                self.settings.segment_length_samples
                + self.settings.extension_padding_samples_start
                + self.settings.extension_padding_samples_end):
            raise ValueError("Input signal too short to create segment.")

        for seg_ind in range((len(self.input_signal) - self.settings.extension_padding_samples_start
                              - self.settings.extension_padding_samples_end)
                             // self.settings.segment_length_samples):
            start_ind = seg_ind*self.settings.segment_length_samples
            end_ind = ((seg_ind + 1)*self.settings.segment_length_samples
                       + self.settings.extension_padding_samples_start + self.settings.extension_padding_samples_end)
            self.segment_list.append(self.input_signal[start_ind:end_ind])

        remaining_samples = len(self.input_signal) - end_ind
        print(f"Input signal split into {seg_ind + 1} segments.")
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
            seg_analysis.analyze_segment()

            seg_analysis.previous_segment = None  # To save storage space when storing in PKL file
            previous_segment = seg_analysis

            self.segment_analysis_list.append(seg_analysis)

        if self.settings.store_csv:
            self.write_results_to_csv()
        if self.settings.store_pkl:
            self.write_result_objects_to_pkl()

    def write_results_to_csv(self):
        """
        Writes the estimated characteristics for all modes in all segments to CSV file.
        :return: None
        """

        # Create new directory if it doesn't already exist
        if not os.path.exists(os.path.dirname(self.results_path_updated)):
            os.makedirs(os.path.dirname(self.results_path_updated))

        # Set unit in [] in header if desired
        if self.settings.unit is None:
            unit_string = ""
        else:
            unit_string = f"[{self.settings.unit}]"

        if self.settings.include_advanced_results:
            dict_keys = ["mode_status",
                         "damping_evaluation",
                         "alarm",
                         "frequency",
                         "median_amp",
                         "damping_ratio",
                         "start_time",
                         "end_time",
                         "init_amp",
                         "final_amp",
                         "freq_start",
                         "freq_stop",
                         "init_amp_est",
                         "decay_rate",
                         "NZF",
                         "interp_frac",
                         "CV"]

            headers = ["Mode status",
                       "Damping evaluation",
                       "Alarm",
                       "Frequency [Hz]",
                       "Median amp. " + unit_string,
                       "Damping ratio",
                       "Start time [s]",
                       "End time [s]",
                       "Init. amp. " + unit_string,
                       "Final amp. " + unit_string,
                       "Frequency start [Hz]",
                       "Frequency stop [Hz]",
                       "Init. amp. est. " + unit_string,
                       "Decay rate",
                       "NZF",
                       "Interp. frac.",
                       "CV"]
        else:
            dict_keys = ["mode_status",
                         "damping_evaluation",
                         "alarm",
                         "frequency",
                         "median_amp",
                         "damping_ratio",
                         "start_time",
                         "end_time"]

            headers = ["Mode status",
                       "Damping evaluation",
                       "Alarm",
                       "Frequency [Hz]",
                       "Median amp. " + unit_string,
                       "Damping ratio",
                       "Start time [s]",
                       "End time [s]"]

        # Adds "_(number)" to file name if permission denied (when file is open in Excel, most likely)
        try:
            with open(self.results_path_updated + ".csv", 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=self.settings.csv_delimiter)

                csv_writer.writerow(["Segment", "Timestamp [s]"] + headers)
                for segment_number, segment in enumerate(self.segment_analysis_list):
                    first_mode_in_segment = True
                    for data_dict in segment.mode_info_list:
                        # Make sure each segment number appears only once, for better readability
                        if first_mode_in_segment:
                            timestamp = (self.settings.extension_padding_time_start
                                         + segment_number*self.settings.segment_length_time)
                            row = [segment_number, timestamp]
                            first_mode_in_segment = False
                        else:
                            row = ["", ""]

                        for key in dict_keys:
                            if isinstance(data_dict[key], float) or isinstance(data_dict[key], np.float64):
                                if key == "damping_ratio" and data_dict["inaccurate_damping_flag"]:
                                    row.append(f"({data_dict[key]:.{self.settings.csv_decimals}f})*")
                                elif key == "frequency" and data_dict["uncertain_mode_flag"]:
                                    row.append(f"({data_dict[key]:.{self.settings.csv_decimals}f})**")
                                else:
                                    row.append(f"{data_dict[key]:.{self.settings.csv_decimals}f}")
                            else:
                                row.append(data_dict[key])
                        csv_writer.writerow(row)

                if self.settings.include_asterisk_explanations:
                    csv_writer.writerow([])
                    csv_writer.writerow(["*Inaccurate damping ratio estimate (high coefficient of variation)."])
                    if not self.settings.skip_storing_uncertain_modes:
                        csv_writer.writerow(["**Uncertain mode (high interpolated fraction)."])

                print(f"Results successfully saved to {self.results_path_updated}.csv.")
        except PermissionError:
            self.file_save_attempt_count += 1
            if self.file_save_attempt_count > 20:
                print("Unable to store results to csv.")
                return
            new_path = self.settings.results_file_path + "_" + str(self.file_save_attempt_count)
            print(
                f"Permission denied for file {self.results_path_updated}.csv. Trying to save to {new_path}.csv instead.")
            self.results_path_updated = new_path

            return self.write_results_to_csv()

    def write_result_objects_to_pkl(self):
        """
        Writes all SegmentAnalysis objects to PKL file.
        :return: None
        """

        # Create new directory if it doesn't already exist
        if not os.path.exists(os.path.dirname(self.results_path_updated)):
            os.makedirs(os.path.dirname(self.results_path_updated))

        try:
            with open(self.results_path_updated + ".pkl", "wb") as file:
                for segment_analysis_obj in self.segment_analysis_list:
                    pickle.dump(segment_analysis_obj, file)
                print(f"Result objects successfully written to {self.results_path_updated}.pkl.")
        except Exception as e:
            print(f"Results were not saved to {self.results_path_updated}.pkl, as the following exception "
                  f"occurred when writing results to pkl file: {e}.")
