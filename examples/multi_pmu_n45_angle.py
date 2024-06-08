"""
Reads from the CSV file with simulated data from the N45 network and analyzes the voltage angle relative to the
reference bus (bus 3000) at 10 PMUs in the N45 network. Summarizes alarms at the end.
"""

import matplotlib.pyplot as plt
import numpy as np

from OscilloWatch.SignalSnapshotAnalysis import SignalSnapshotAnalysis
from OscilloWatch.OWSettings import OWSettings
from OscilloWatch.csv_column_to_list import csv_column_to_list
from OscilloWatch.post_processing import summarize_alarms


fs = 50
file_path = "pmu_data_csv/Generator Loss event N45.CSV"


reference_angle = np.unwrap(np.array(csv_column_to_list(file_path, 2)), period=360)  # b3000

t = np.linspace(0, len(reference_angle)/fs, len(reference_angle))
plt.figure()
plt.xlabel("Time [s]")
plt.ylabel("Voltage angle [deg]")


settings = OWSettings(
    fs=fs,
    segment_length_time=10,
    extension_padding_time_start=10,
    extension_padding_time_end=2,
    csv_delimiter=",",  # Change to ";" if your Excel interprets "," as decimal separator
    minimum_amplitude=0.2,
    alarm_median_amplitude_threshold=1.0,
    minimum_frequency=0.1,
    include_asterisk_explanations=False,
    unit="deg"
)


settings.results_file_path = "results/N45/NO1"
b5120 = (np.unwrap(np.array(csv_column_to_list(file_path, 32, delimiter=",")), period=360)
         - reference_angle + np.random.normal(0, .05, len(reference_angle)))
snap_an = SignalSnapshotAnalysis(b5120, settings)
snap_an.analyze_whole_signal()
plt.plot(t, b5120, label="NO1")

settings.results_file_path = "results/N45/NO2"
b5240 = (np.unwrap(np.array(csv_column_to_list(file_path, 47, delimiter=",")), period=360)
         - reference_angle + np.random.normal(0, .05, len(reference_angle)))
snap_an = SignalSnapshotAnalysis(b5240, settings)
snap_an.analyze_whole_signal()
plt.plot(t, b5240, label="NO2")

settings.results_file_path = "results/N45/NO3"
b5320 = (np.unwrap(np.array(csv_column_to_list(file_path, 62, delimiter=",")), period=360)
         - reference_angle + np.random.normal(0, .05, len(reference_angle)))
snap_an = SignalSnapshotAnalysis(b5320, settings)
snap_an.analyze_whole_signal()
plt.plot(t, b5320, label="NO3")

settings.results_file_path = "results/N45/NO4"
b5420 = (np.unwrap(np.array(csv_column_to_list(file_path, 71, delimiter=",")), period=360)
         - reference_angle + np.random.normal(0, .05, len(reference_angle)))
snap_an = SignalSnapshotAnalysis(b5420, settings)
snap_an.analyze_whole_signal()
plt.plot(t, b5420, label="NO4")

settings.results_file_path = "results/N45/NO5"
b5560 = (np.unwrap(np.array(csv_column_to_list(file_path, 88, delimiter=",")), period=360)
         - reference_angle + np.random.normal(0, .05, len(reference_angle)))
snap_an = SignalSnapshotAnalysis(b5560, settings)
snap_an.analyze_whole_signal()
plt.plot(t, b5560, label="NO5")

settings.results_file_path = "results/N45/SE1"
b3115 = (np.unwrap(np.array(csv_column_to_list(file_path, 9, delimiter=",")), period=360)
         - reference_angle + np.random.normal(0, .05, len(reference_angle)))
snap_an = SignalSnapshotAnalysis(b3115, settings)
snap_an.analyze_whole_signal()
plt.plot(t, b3115, label="SE1")

settings.results_file_path = "results/N45/SE2"
b3245 = (np.unwrap(np.array(csv_column_to_list(file_path, 14, delimiter=",")), period=360)
         - reference_angle + np.random.normal(0, .05, len(reference_angle)))
snap_an = SignalSnapshotAnalysis(b3245, settings)
snap_an.analyze_whole_signal()
plt.plot(t, b3245, label="SE2")

settings.results_file_path = "results/N45/SE3"
b3359 = (np.unwrap(np.array(csv_column_to_list(file_path, 23, delimiter=",")), period=360)
         - reference_angle + np.random.normal(0, .05, len(reference_angle)))
snap_an = SignalSnapshotAnalysis(b3359, settings)
snap_an.analyze_whole_signal()
plt.plot(t, b3359, label="SE3")

settings.results_file_path = "results/N45/SE4"
b8500 = (np.unwrap(np.array(csv_column_to_list(file_path, 101, delimiter=",")), period=360)
         - reference_angle + np.random.normal(0, .05, len(reference_angle)))
snap_an = SignalSnapshotAnalysis(b8500, settings)
snap_an.analyze_whole_signal()
plt.plot(t, b8500, label="SE4")

settings.results_file_path = "results/N45/FI"
b7000 = (np.unwrap(np.array(csv_column_to_list(file_path, 91, delimiter=",")), period=360)
         - reference_angle + np.random.normal(0, .05, len(reference_angle)))
snap_an = SignalSnapshotAnalysis(b7000, settings)
snap_an.analyze_whole_signal()
plt.plot(t, b7000, label="FI")

try:
    summarize_alarms([
        ("results/N45/NO1.pkl", "NO1"),
        ("results/N45/NO2.pkl", "NO2"),
        ("results/N45/NO3.pkl", "NO3"),
        ("results/N45/NO4.pkl", "NO4"),
        ("results/N45/NO5.pkl", "NO5"),
        ("results/N45/SE1.pkl", "SE1"),
        ("results/N45/SE2.pkl", "SE2"),
        ("results/N45/SE3.pkl", "SE3"),
        ("results/N45/SE4.pkl", "SE4"),
        ("results/N45/FI.pkl", "FI"),
    ],
        csv_delimiter=",",
        results_file_path="results/N45/alarms.csv"
    )
except PermissionError:
    print("Permission denied for alarm summary file.")

plt.legend(loc="lower right")
plt.show()
