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


        # Initialize PDC
        self.pdc = Pdc(pdc_id=self.settings.pdc_id, pmu_ip=self.settings.pmu_ip, pmu_port=self.settings.pmu_port)
        #self.pdc.logger.setLevel("DEBUG")
        self.pdc.run()  # Connect to PMU
        self.pmu_config = self.pdc.get_config()  # Get configuration from PMU
        self.pmu_header = self.pdc.get_header()

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
                # df["measurements"][pmu_index]["phasors"][phasor_index][0 for magnitude, 1 for angle]
                values_segment = np.array([df["measurements"][0]["phasors"][0][0] for df in df_segment])

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
                            extension_padding_time_end=1)
rta = RealTimeAnalysis(settings)
rta.run_analysis()
