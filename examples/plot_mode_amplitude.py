import numpy as np
import matplotlib.pyplot as plt

from OscilloWatch.post_processing.get_mode_amplitude_evolution import get_mode_amplitude_evolution
from OscilloWatch.post_processing.read_from_pkl import read_from_pkl

mode_freq = 0.4

seg_lst = read_from_pkl("../results/results.pkl")
amplitude_curve = get_mode_amplitude_evolution(seg_lst, mode_freq)

tAxis = np.linspace(0,
                    (seg_lst[0].settings.extension_padding_time_start
                     + seg_lst[0].settings.segment_length_time * len(seg_lst)),
                    len(amplitude_curve))

plt.figure()
plt.title(f"Amplitude of {mode_freq} Hz mode")
plt.plot(tAxis, amplitude_curve)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()
