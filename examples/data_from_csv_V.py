"""
Reads from the CSV file with simulated data from the N45 network and analyzes the voltage magnitude at bus 3000 in the
N45 network.
"""

import matplotlib.pyplot as plt
import numpy as np

from OscilloWatch.SignalSnapshotAnalysis import SignalSnapshotAnalysis
from OscilloWatch.OWSettings import OWSettings
from OscilloWatch.csv_column_to_list import csv_column_to_list


file_path = "pmu_data_csv/Generator Loss event N45.CSV"
column_index = 1
fs = 50

data = csv_column_to_list(file_path, column_index)
t = np.linspace(0, len(data)/fs, len(data))
plt.figure()
plt.plot(t, data)
plt.xlabel("Time [s]")
plt.ylabel("Voltage magnitude [pu]")
#plt.show()

settings = OWSettings(
    fs=fs,
    segment_length_time=10,
    extension_padding_time_start=10,
    extension_padding_time_end=2,
    csv_delimiter=",",  # Change to ";" if your Excel interprets "," as decimal separator
    minimum_amplitude=0.001,
    minimum_frequency=0.1,
    results_file_path="results/N45_b3000_V",
    unit="pu"
)

snap_an = SignalSnapshotAnalysis(data, settings)
snap_an.analyze_whole_signal()

plt.show()
