"""
Reads from the CSV file with simulated data from the N45 network and analyzes the voltage angle relative to the
reference bus (bus 3000) at bus 5110 in the N45 network.
"""

import matplotlib.pyplot as plt
import numpy as np

from OscilloWatch.SignalSnapshotAnalysis import SignalSnapshotAnalysis
from OscilloWatch.OWSettings import OWSettings
from OscilloWatch.csv_column_to_list import csv_column_to_list

file_path = "pmu_data_csv/Generator Loss event N45.CSV"

# Uses bus 3000 as reference angle
reference_angle = np.unwrap(np.array(csv_column_to_list(file_path, 2)), period=360)

fs = 50

# Measure at bus 5110
angle_data = (np.unwrap(np.array(csv_column_to_list(file_path, 30)), period=360)
              - reference_angle)

# Create plot
t = np.linspace(0, len(angle_data)/fs, len(angle_data))
plt.figure()
plt.plot(t, angle_data)
plt.xlabel("Time [s]")
plt.ylabel("Voltage angle [deg]")
#plt.show()

settings = OWSettings(
    fs=fs,
    segment_length_time=10,
    extension_padding_time_start=10,
    extension_padding_time_end=2,
    print_emd_sifting_details=False,
    csv_delimiter=",",  # Change to ";" if your Excel interprets "," as decimal separator
    minimum_amplitude=0.01,
    alarm_median_amplitude_threshold=1.0,
    minimum_frequency=0.1,
    results_file_path="results/N45_b5110_angle",
    unit="deg"
)

snap_an = SignalSnapshotAnalysis(angle_data, settings)
snap_an.analyze_whole_signal()

plt.show()
