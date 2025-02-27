"""
This script creates a summary of the alarms raised from multiple analyzed PMU signals.
The script "multi_pmu_n45_angle.py" should be run before this one.
"""

from OscilloWatch.post_processing import summarize_alarms

summarize_alarms(
    [
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
    csv_delimiter=",",  # Change to ";" if your Excel interprets "," as decimal separator
    results_file_path="results/N45/alarms.csv"
)
