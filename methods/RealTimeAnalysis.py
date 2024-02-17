import csv
import pickle
import threading
import datetime

import numpy as np
import matplotlib.pyplot as plt
from synchrophasor.pdc import Pdc
from synchrophasor.frame import DataFrame

from methods.AnalysisSettings import AnalysisSettings
from methods.SegmentAnalysis import SegmentAnalysis


class RealTimeAnalysis:

    def __init__(self, settings: AnalysisSettings):
        self.settings = settings
        self.df_buffer = []
        self.segment_number_csv = 0
        self.segment_number_pkl = 0

        self.result_buffer_csv = []
        self.result_buffer_pkl = []
        self.csv_path_full = self.settings.results_file_path + ".csv"
        self.csv_open_attempts = 0

        print(f"Attempting to connect to {self.settings.ip}:{self.settings.port} (Device ID: "
              f"{self.settings.sender_device_id})")
        # Initialize PDC
        self.pdc = Pdc(pdc_id=self.settings.sender_device_id, pmu_ip=self.settings.ip, pmu_port=self.settings.port)
        #self.pdc.logger.setLevel("DEBUG")
        self.pdc.run()  # Connect to PMU

        #self.pdc.stop()
        self.pmu_config = self.pdc.get_config()  # Get configuration from PMU
        #self.pmu_header = self.pdc.get_header()  # Get header from PMU

        # Initialize indices before finding the correct values
        self.id_index = 0
        self.channel_index = 0
        self.component_index = 0
        self.find_indices()

        self.init_csv()  # Clear existing csv file or create new

        with open(self.settings.results_file_path + ".pkl", "wb") as file:
            # Clears existing pkl file or creates new:
            print(f"{self.settings.results_file_path}.pkl will be used for storing segment result objects.")

    def find_indices(self):
        if isinstance(self.pmu_config.get_stream_id_code(), int):
            if self.pmu_config.get_stream_id_code() != self.settings.pmu_id:
                raise ValueError(f"{self.settings.pmu_id} is not a valid PMU ID code.")
            # Create list of phasor channel names, with spaces at the end of the strings removed
            phasor_channel_names = []
            for channel_name in self.pmu_config.get_channel_names()[:self.pmu_config.get_phasor_num()]:
                for i in range(len(channel_name) - 1, 0, -1):
                    if channel_name[i] == " ":
                        channel_name = channel_name[:i]
                    else:
                        phasor_channel_names.append(channel_name)
                        break

        elif isinstance(self.pmu_config.get_stream_id_code(), list):
            # Find index of PMU ID, use to create list of phasor channel names from the PMU, with spaces at the end of
            # the strings removed
            if self.settings.pmu_id not in self.pmu_config.get_stream_id_code():
                raise ValueError(f"{self.settings.pmu_id} is not a valid PMU ID code.")
            self.id_index = self.pmu_config.get_stream_id_code().index(self.settings.pmu_id)
            phasor_channel_names = []
            for channel_name in (self.pmu_config.get_channel_names()
                                 [self.id_index][:self.pdc.pmu_cfg2.get_phasor_num()[self.id_index]]):
                for i in range(len(channel_name) - 1, 0, -1):
                    if channel_name[i] == " ":
                        channel_name = channel_name[:i]
                    else:
                        phasor_channel_names.append(channel_name)
                        break
        else:
            raise TypeError("Invalid type of ID code in config frame. Must be int or list")

        if self.settings.channel.lower() != "freq" and self.settings.channel.lower() != "frequency":
            if self.settings.channel not in phasor_channel_names:
                raise ValueError(f"{self.settings.channel} is not a valid channel name.")
            self.channel_index = phasor_channel_names.index(self.settings.channel)

        if self.settings.phasor_component.lower() == "magnitude":
            self.component_index = 0
        elif self.settings.phasor_component.lower() == "angle":
            self.component_index = 1
        else:
            raise ValueError(f"{self.settings.phasor_component} is not a valid phasor component. Must be 'magnitude' "
                             f"or 'angle'.")

        self.settings.fs = self.pmu_config.get_data_rate()
        self.settings.update_calc_values()

    def receive_data_frames(self):
        df_count = 0
        while True:
            #print(df_count)
            data = self.pdc.get()  # Keep receiving data

            if isinstance(data, DataFrame):
                self.df_buffer.append(data.get_measurements())
            else:
                if not data:
                    self.pdc.quit()  # Close connection
                    break
                if len(data) > 0:
                    for meas in data:
                        self.df_buffer.append(meas.get_measurements())
            df_count += 1

    def run_analysis(self):
        self.pdc.start()  # Request connected PMU to start sending measurements

        # Start continuously receiving data and adding to buffer in own thread, so no data is lost if it is sent while
        # the main loop is running.
        receive_thread = threading.Thread(target=self.receive_data_frames)
        receive_thread.start()

        # Start main processing loop
        while True:
            # If buffer has enough samples to create segment:
            if len(self.df_buffer) >= self.settings.total_segment_length_samples:
                # Create segment of data frames, remove correct amount of samples from start of buffer
                df_segment = self.df_buffer[:self.settings.total_segment_length_samples]
                self.df_buffer = self.df_buffer[(self.settings.segment_length_samples
                                                + self.settings.extension_padding_samples_end):]

                # Create segment array with wanted numerical values found from dataframe using dict keys and indices
                if self.settings.channel.lower() == "freq" or self.settings.channel.lower() == "frequency":
                    values_segment = np.array([df["measurements"][self.id_index]["frequency"] for df in df_segment])
                else:
                    values_segment = np.array([df["measurements"][self.id_index]["phasors"][self.channel_index]
                                               [self.component_index] for df in df_segment])
                timestamp = df_segment[self.settings.extension_padding_samples_start]["time"]  # Epoch time
                timestamp_str = datetime.datetime.fromtimestamp(timestamp)

                #plt.figure()
                #plt.plot(values_segment)

                seg_an = SegmentAnalysis(values_segment, self.settings, timestamp_str)
                seg_an.damping_analysis()
                self.result_buffer_csv.append(seg_an)
                self.result_buffer_pkl.append(seg_an)
                self.add_segment_result_to_csv()
                self.add_segment_result_to_pkl()
                #seg_an.hht.emd.plot_emd_results(show=False)
                #seg_an.hht.plot_hilbert_spectrum(show=False)

                #plt.show()

    def init_csv(self, file_path="default"):
        headers = list(self.settings.blank_mode_info_dict)

        if file_path == "default":
            current_file_path = self.csv_path_full
        else:
            current_file_path = file_path

        # Adds "_(number)" to file name if permission denied (when file is open in Excel, most likely)
        try:
            with open(current_file_path, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=self.settings.csv_delimiter)
                csv_writer.writerow(["Segment", "Timestamp"] + headers)
                self.csv_path_full = current_file_path
                print(f"{current_file_path} will be used for storing numerical results.")
        except PermissionError:
            self.csv_open_attempts += 1
            if self.csv_open_attempts > 20:
                print("File opening attempts exceeded limit. Unable to save results to csv.")
                return
            new_path = self.settings.results_file_path + "_" + str(self.csv_open_attempts) + ".csv"
            print(f"Permission denied for file {current_file_path}. Trying to save to {new_path} instead.")

            return self.init_csv(new_path)

    def add_segment_result_to_csv(self):
        headers = list(self.settings.blank_mode_info_dict)
        try:
            with open(self.csv_path_full, 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=self.settings.csv_delimiter)
                while self.result_buffer_csv:
                    first_mode_in_segment = True
                    for data_dict in self.result_buffer_csv[0].mode_info_list:
                        # Make sure each segment number appears only once, for better readability
                        if first_mode_in_segment:
                            row = [self.segment_number_csv, self.result_buffer_csv[0].timestamp]
                            first_mode_in_segment = False
                        else:
                            row = ["", ""]

                        for header in headers:
                            if isinstance(data_dict[header], float) or isinstance(data_dict[header], np.float64):
                                row.append(f"{data_dict[header]:.{self.settings.csv_decimals}f}")
                            else:
                                row.append(data_dict[header])
                        csv_writer.writerow(row)
                    print(f"Results for segment {self.segment_number_csv} added to {self.csv_path_full}.")
                    self.segment_number_csv += 1
                    del self.result_buffer_csv[0]
        except Exception as e:
            print(f"Exception during csv storing: {e}. Attempting to store again after the next segment is analyzed.")

    def add_segment_result_to_pkl(self):
        try:
            with open(self.settings.results_file_path + ".pkl", 'ab') as file:
                while self.result_buffer_pkl:
                    pickle.dump(self.result_buffer_pkl[0], file)
                    print(f"Results for segment {self.segment_number_pkl} added to "
                          f"{self.settings.results_file_path}.pkl.")
                    self.segment_number_pkl += 1
                    del self.result_buffer_pkl[0]
        except Exception as e:
            print(f"Exception during pkl storing: {e}. Attempting to store again after the next segment is analyzed.")

# Todo: Match pkl and csv filenames
# Todo: Print segment numbers