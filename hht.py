import numpy as np
import matplotlib.pyplot as plt
from emd import emd
from scipy.signal import hilbert

def hilbert_spectrum_single(signal):
    hilbert_signal = hilbert(signal)
    theta = np.gradient(np.angle(hilbert_signal))
    plt.figure()
    plt.plot(theta)
    plt.show()


def f(t):
    return 10*np.exp(.2*t)*np.cos(2.4*np.pi*t) + 8*np.exp(-.1*t)*np.cos(np.pi*t)

start = 0
end = 5
fs = 1000
input_signal = f(np.arange(start, end, 1/fs))

#Random (reproducable) signal
np.random.seed(0)
#input_signal = np.random.randn(500)

#input_signal_derivative = np.gradient(input_signal)

#plt.figure()
#plt.plot(input_signal)
#plt.show()




hilbert_spectrum_single(input_signal)