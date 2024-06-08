import matplotlib.pyplot as plt
import numpy as np

from OscilloWatch.post_processing import reconstruct_signal
from OscilloWatch.post_processing import read_from_pkl

seg_res_list = read_from_pkl("../results/N45/NO2.pkl")
signal = reconstruct_signal(seg_res_list)

fs = seg_res_list[0].settings.fs
t = np.linspace(0, len(signal)/fs, len(signal))

plt.figure()
plt.plot(t, signal)
plt.xlabel("Time [s]")
plt.ylabel("Voltage [pu]")
plt.tight_layout()
plt.show()
