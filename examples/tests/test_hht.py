import numpy as np
import matplotlib.pyplot as plt
from OscilloWatch.HHT import HHT, moving_average
from OscilloWatch.EMD import EMD
from OscilloWatch.OWSettings import OWSettings

np.random.seed(0)

def f_old(t):
    #return 6*np.exp(.2*t)*np.cos(3*np.pi*t) + 15*np.exp(-.1*t)*np.cos(np.pi*t)
    return (10*np.exp(.2*t)*np.cos(2.4*np.pi*t)
            #+ 8*np.exp(-.1*t)*np.cos(np.pi*t)
            #+ 3*np.exp(.3*t)*np.cos(5*np.pi*t)
            + 20*np.exp(-.2*t)*np.cos(10*np.pi*t))

def g(t):
    return f(t) + 10*np.exp(-.2*t)*np.cos(2.6*np.pi*t)

def a(t):
    return 4*np.exp(.1*t)

def f(t):
    return a(t)*np.cos(3*2*np.pi*t)


settings = OWSettings(max_imfs=4, extension_padding_time_start=5, extension_padding_time_end=10)
start = -5
end = 20
fs = 50

t = np.arange(start, end, 1/fs)
input_signal1 = g(t)# + np.random.normal(0, 5, (end-start)*fs)
# Random (reproducable) signal
input_signal2 = np.random.randn(500)

#input_signal1 = moving_average(input_signal1, settings1.noise_reduction_moving_avg_window)


hht = HHT(input_signal1, settings)
hht.full_hht()
hht.emd.plot_emd_results()
hht.plot_hilbert_spectrum(show=False)

plt.show()