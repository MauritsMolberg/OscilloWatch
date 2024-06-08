import numpy as np
import matplotlib.pyplot as plt

from OscilloWatch.HHT import HHT
from OscilloWatch.OWSettings import OWSettings


def f(t):
    return (
        20*np.exp(-.2*t)*np.cos(10*np.pi*t)
        + 8*np.exp(.1*t)*np.cos(5*np.pi*t)
        + 10*np.exp(.15*t)*np.cos(2.4*np.pi*t)
        + 16*np.exp(.1)*np.cos(np.pi*t)
    )


start = -10
end = 20
fs = 50
t = np.arange(start, end, 1/fs)

input_signal1 = f(t)

settings = OWSettings(
    extension_padding_time_start=10,
    extension_padding_time_end=10
)

hht = HHT(input_signal1, settings)
hht.full_hht()
#hht.calc_hilbert_spectrum(input_signal1)  # Calculate spectrum directly without EMD
hht.emd.plot_emd_results()
hht.plot_hilbert_spectrum(show=False)

plt.show()
