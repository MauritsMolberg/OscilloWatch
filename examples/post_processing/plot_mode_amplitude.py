"""
This script creates a plot of how the amplitude of a mode changes over time, at multiple locations (PMUs).
The script "multi_pmu_n45_angle.py" should be run before this one.
"""

import numpy as np
import matplotlib.pyplot as plt

from OscilloWatch.post_processing import get_mode_amplitude_evolution
from OscilloWatch.post_processing import read_from_pkl


mode_freq = .85  # Edit this value to choose which mode to investigate.
freq_tolerance = 0.15

SE1 = get_mode_amplitude_evolution(read_from_pkl("../results/N45/SE1.pkl"), mode_frequency=mode_freq, tolerance=freq_tolerance)
SE2 = get_mode_amplitude_evolution(read_from_pkl("../results/N45/SE2.pkl"), mode_frequency=mode_freq, tolerance=freq_tolerance)
SE3 = get_mode_amplitude_evolution(read_from_pkl("../results/N45/SE3.pkl"), mode_frequency=mode_freq, tolerance=freq_tolerance)
SE4 = get_mode_amplitude_evolution(read_from_pkl("../results/N45/SE4.pkl"), mode_frequency=mode_freq, tolerance=freq_tolerance)
NO1 = get_mode_amplitude_evolution(read_from_pkl("../results/N45/NO1.pkl"), mode_frequency=mode_freq, tolerance=freq_tolerance)
NO2 = get_mode_amplitude_evolution(read_from_pkl("../results/N45/NO2.pkl"), mode_frequency=mode_freq, tolerance=freq_tolerance)
NO3 = get_mode_amplitude_evolution(read_from_pkl("../results/N45/NO3.pkl"), mode_frequency=mode_freq, tolerance=freq_tolerance)
NO4 = get_mode_amplitude_evolution(read_from_pkl("../results/N45/NO4.pkl"), mode_frequency=mode_freq, tolerance=freq_tolerance)
NO5 = get_mode_amplitude_evolution(read_from_pkl("../results/N45/NO5.pkl"), mode_frequency=mode_freq, tolerance=freq_tolerance)
FI = get_mode_amplitude_evolution(read_from_pkl("../results/N45/FI.pkl"), mode_frequency=mode_freq, tolerance=freq_tolerance)

seg_lst = read_from_pkl("../results/N45/SE1.pkl")
t = np.linspace(0,
                seg_lst[0].settings.segment_length_time*len(seg_lst),
                max(len(SE1), len(SE2), len(SE3), len(SE4), len(NO1), len(NO2), len(NO3), len(NO4), len(NO5), len(FI)))

plt.figure()
plt.plot(t, NO1, label="NO1")
plt.plot(t, NO2, label="NO2")
plt.plot(t, NO3, label="NO3")
plt.plot(t, NO4, label="NO4")
plt.plot(t, NO5, label="NO5")
plt.plot(t, SE1, label="SE1")
plt.plot(t, SE2, label="SE2")
plt.plot(t, SE3, label="SE3")
plt.plot(t, SE4, label="SE4")
plt.plot(t, FI, label="FI")

plt.xlabel("Time [s]")
plt.ylabel("Amplitude [deg]")
plt.legend(loc="upper right")
plt.tight_layout()
#plt.savefig(f"{mode_freq}Hz_amplitude.png", dpi=1000)
plt.show()

