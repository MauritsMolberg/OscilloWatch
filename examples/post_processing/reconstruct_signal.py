from OscilloWatch.post_processing import reconstruct_signal
from OscilloWatch.post_processing import read_from_pkl
import matplotlib.pyplot as plt
import numpy as np

seg_res_list = read_from_pkl("../../results/N45/gen/b7000.pkl")
signal = reconstruct_signal(seg_res_list)

fs = seg_res_list[0].settings.fs
t = np.linspace(0, len(signal)/fs, len(signal))

plt.figure()
plt.plot(t, signal)
plt.show()
