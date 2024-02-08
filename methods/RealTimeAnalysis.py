import numpy as np
import csv
import os
from methods.AnalysisSettings import AnalysisSettings
from methods.SegmentAnalysis import SegmentAnalysis
from synchrophasor.pdc import Pdc
from synchrophasor.frame import DataFrame
import matplotlib.pyplot as plt

class RealTimeAnalysis:

    def __init__(self, settings : AnalysisSettings):
        self.settings = settings
        self.df_buffer = []

        self.segment_analysis_list = []

        print(f"Attempting to connect to {self.settings.ip}:{self.settings.port} (Device ID: {self.settings.pmu_id})")
        # Initialize PDC
        self.pdc = Pdc(pdc_id=self.settings.device_id, pmu_ip=self.settings.ip, pmu_port=self.settings.port)
        #self.pdc.logger.setLevel("DEBUG")
        self.pdc.run()  # Connect to PMU

        #self.pdc.stop()
        self.pmu_config = self.pdc.get_config()  # Get configuration from PMU
        #self.pmu_header = self.pdc.get_header()  # Get header from PMU

        if type(self.pmu_config._id_code) == int:
            if self.pmu_config._id_code != self.settings.pmu_id:
                raise ValueError(f"{self.settings.pmu_id} is not a valid PMU ID code.")
            self.id_index = 0
            # Create list of phasor channel names, with spaces at the end of the strings removed
            phasor_channel_names = []
            for channel_name in self.pmu_config._channel_names[:self.pdc.pmu_cfg2._phasor_num]:
                for i in range(len(channel_name) - 1, 0, -1):
                    if channel_name[i] == " ":
                        channel_name = channel_name[:i]
                    else:
                        phasor_channel_names.append(channel_name)
                        break

        elif type(self.pmu_config._id_code) == list:
            # Find index of PMU ID, use to create list of phasor channel names from the PMU, with spaces at the end of
            # the strings removed
            if not self.settings.pmu_id in self.pmu_config._id_code:
                raise ValueError(f"{self.settings.pmu_id} is not a valid PMU ID code.")
            self.id_index = self.pmu_config._id_code.index(self.settings.pmu_id)

            phasor_channel_names = []
            for channel_name in (self.pmu_config._channel_names
                                 [self.id_index][:self.pdc.pmu_cfg2._phasor_num[self.id_index]]):
                for i in range(len(channel_name) - 1, 0, -1):
                    if channel_name[i] == " ":
                        channel_name = channel_name[:i]
                    else:
                        phasor_channel_names.append(channel_name)
                        break
        else:
            raise TypeError("Invalid type of ID code in config frame. Must be int or list")

        # Find correct indices for PMU and channel in data frame, based on info from config frame
        if self.settings.channel.lower() != "freq" and self.settings.channel.lower() != "frequency":
            if not self.settings.channel in phasor_channel_names:
                raise ValueError(f"{self.settings.channel} is not a valid channel name.")
            self.channel_index = phasor_channel_names.index(self.settings.channel)

        if self.settings.phasor_component.lower() == "magnitude":
            self.component_index = 0
        elif self.settings.phasor_component.lower() == "angle":
            self.component_index = 1
        else:
            raise ValueError(f"{self.settings.phasor_component} is not a valid phasor component. Must be 'magnitude'"
                             f"or 'angle'.")

    def run_analysis(self):
        self.pdc.start()  # Request connected PMU to start sending measurements

        df_count = 0
        while True:
            print(df_count)
            data = self.pdc.get()  # Keep receiving data

            if type(data) == DataFrame:
                self.df_buffer.append(data.get_measurements())
            else:
                if not data:
                    self.pdc.quit()  # Close connection
                    break
                if len(data) > 0:
                    for meas in data:
                        self.df_buffer.append(meas.get_measurements())
            df_count += 1
            if len(self.df_buffer) >= self.settings.total_segment_length_samples:  # Enough samples to analyze segment
                # Create segment, remove correct amount of samples from start of buffer
                df_segment = self.df_buffer[:self.settings.total_segment_length_samples]
                self.df_buffer = self.df_buffer[(self.settings.segment_length_samples
                                                + self.settings.extension_padding_samples_end):]

                # Create array of numerical values found from dataframe using dict keys and indices
                if self.settings.channel.lower() == "freq" or self.settings.channel.lower() == "frequency":
                    values_segment = np.array([df["measurements"][self.id_index]["frequency"] for df in df_segment])
                else:
                    values_segment = np.array([df["measurements"][self.id_index]["phasors"][self.channel_index]
                                               [self.component_index] for df in df_segment])

                plt.figure()
                plt.plot(values_segment)

                seg_an = SegmentAnalysis(values_segment, self.settings)
                seg_an.damping_analysis()
                seg_an.hht.emd.plot_emd_results(show=False)
                seg_an.hht.plot_hilbert_spectrum(show=False)

                plt.show()


settings = AnalysisSettings(fs=50,
                            segment_length_time=5,
                            extension_padding_time_start=2,
                            extension_padding_time_end=1,
                            channel="signal",
                            #ip="10.100.0.75",
                            #port=34702,
                            device_id=45,
                            pmu_id=1410
                            )
rta = RealTimeAnalysis(settings)
rta.run_analysis()
