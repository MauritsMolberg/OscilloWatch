import numpy as np
import matplotlib.pyplot as plt

from OscilloWatch.get_mode_amplitude_evolution import get_mode_amplitude_evolution
from OscilloWatch.read_from_pkl import read_from_pkl

mode_freq = 3.3

WE = get_mode_amplitude_evolution(read_from_pkl("../results/WE.pkl"), mode_freq, fs=50)
SE = get_mode_amplitude_evolution(read_from_pkl("../results/SE.pkl"), mode_freq, fs=50)
MD = get_mode_amplitude_evolution(read_from_pkl("../results/MD.pkl"), mode_freq, fs=50)
SO = get_mode_amplitude_evolution(read_from_pkl("../results/SO.pkl"), mode_freq, fs=50)
NW = get_mode_amplitude_evolution(read_from_pkl("../results/NW.pkl"), mode_freq, fs=50)
SW = get_mode_amplitude_evolution(read_from_pkl("../results/SW.pkl"), mode_freq, fs=50)
NO = get_mode_amplitude_evolution(read_from_pkl("../results/NO.pkl"), mode_freq, fs=50)
NE = get_mode_amplitude_evolution(read_from_pkl("../results/NE.pkl"), mode_freq, fs=50)

seg_lst = read_from_pkl("../results/WE.pkl")
tAxis_amp = np.linspace(0,
                        seg_lst[0].settings.segment_length_time*len(seg_lst),
                        max(len(WE), len(SE), len(MD), len(SO), len(NW), len(SW), len(NO), len(NE)))

# full_signal = csv_column_to_list("../example_pmu_data/180924-osc-frekvens-og-vinkel_semicolon.csv", 7, delimiter=";")
# tAxis = np.arange(0, len(full_signal)/seg_lst[0].settings.fs, 1/seg_lst[0].settings.fs)

# plt.figure()
# plt.title("Time series plot")
# plt.plot(tAxis, full_signal)
# #plt.ylabel("Active power")
# plt.xlabel("Time [s]")

plt.figure()
#plt.title(f"Amplitude of {mode_freq} Hz mode")

plt.plot(tAxis_amp, NE, label="NE")
plt.plot(tAxis_amp, NO, label="NO")
plt.plot(tAxis_amp, NW, label="NW")
plt.plot(tAxis_amp, MD, label="MD")
plt.plot(tAxis_amp, WE, label="WE")
plt.plot(tAxis_amp, SW, label="SW")
plt.plot(tAxis_amp, SO, label="SO")
plt.plot(tAxis_amp, SE, label="SE")

plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend(loc="upper right")
plt.axis([0,600,0,0.026])
plt.tight_layout()
plt.show()