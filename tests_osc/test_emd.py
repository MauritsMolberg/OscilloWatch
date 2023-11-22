import numpy as np
import matplotlib.pyplot as plt
from methods.HHT import HHT, moving_average
from methods.EMD import EMD
from methods.AnalysisSettings import AnalysisSettings

np.random.seed(0)

def f(t):
    #return 6*np.exp(.2*t)*np.cos(3*np.pi*t) + 15*np.exp(-.1*t)*np.cos(np.pi*t)
    return 10*np.exp(.2*t)*np.cos(2.4*np.pi*t)
            #+ 8*np.exp(-.1*t)*np.cos(np.pi*t)
            #+ 3*np.exp(.3*t)*np.cos(5*np.pi*t)
            #+ 20*np.exp(-.2*t)*np.cos(10*np.pi*t))

def g(t):
    return 4*np.exp(.2*t)*np.cos(2.6*np.pi*t)



settings = AnalysisSettings(remove_padding_after_emd=False, max_imfs=3)

start = 0
end = 11
fs = 50
t = np.arange(start, end, 1/fs)
input_signal1 = g(t)
# Random (reproducable) signal
input_signal2 = np.random.randn(500)


emd = EMD(input_signal1, settings)
emd.perform_emd()
emd.plot_emd_results(show=False)

plt.show()