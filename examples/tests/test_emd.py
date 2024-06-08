import numpy as np
import matplotlib.pyplot as plt

from OscilloWatch.EMD import EMD
from OscilloWatch.OWSettings import OWSettings


def f(t):
    return (
        20*np.exp(-.2*t)*np.cos(10*np.pi*t)
        + 8*np.exp(.1*t)*np.cos(5*np.pi*t)
        + 10*np.exp(.15*t)*np.cos(2.4*np.pi*t)
        + 16*np.exp(.1)*np.cos(np.pi*t)
    )


start = 0
end = 10
fs = 50
t = np.arange(start, end, 1/fs)

input_signal = f(t)

settings = OWSettings()

emd = EMD(input_signal, settings)
emd.perform_emd()
emd.plot_emd_results(show=False)

plt.show()
