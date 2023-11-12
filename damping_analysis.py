import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from emd import emd, plot_emd_results
from hht import hht, plot_hilbert_spectrum

# Exponential decay model function
def exponential_decay_model(t, A, k):
    return A * np.exp(-k * t)

def damped_sinusoidal_model(t, A, f, zeta, phi):
    return A * np.exp(-zeta * 2*np.pi*f * t) * np.sin(2*np.pi*f * t + phi)


def plot_damped_sinusoidal(t, A, f, zeta, phi):
    plt.figure()


def damping_analysis(hilbert_spectrum):
    damping_info = []
    time_points = np.arange(len(hilbert_spectrum[0]))

    # Extract parameters for each frequency component
    for row in hilbert_spectrum:
        popt, _ = curve_fit(damped_sinusoidal_model, time_points, row)
        damping_info.append(popt)  # popt contains [A, k]

    return damping_info




def f(t):
    return 10*np.exp(.2*t)*np.cos(2.4*np.pi*t)# + 8*np.exp(-.1*t)*np.cos(np.pi*t) # 1.2 Hz (growing) + 0.5 Hz (decaying)
    #return (10*np.exp(.2*t)*np.cos(2.4*np.pi*t)
            #+ 8*np.exp(-.1*t)*np.cos(np.pi*t)
            #+ 2*np.exp(.3*t)*np.cos(24*np.pi*t)
            #+ 14*np.exp(-.2*t)*np.cos(10*np.pi*t))
    #return 3*np.sin(5*np.pi*t)

start = 0
end = 10
fs = 50
input_signal1 = f(np.arange(start, end, 1/fs))

# Random (reproducable) signal
np.random.seed(0)
input_signal2 = np.random.randn(500)


imf_list, res = emd(input_signal1, remove_padding=True)

plot_emd_results(input_signal1, imf_list, res, fs, show=False)

damping_info = damping_analysis(imf_list)

hilbert_spectrum, freqAxis = hht(input_signal1,
                                max_imfs=6,
                                sampling_freq=fs,
                                mirror_padding_fraction=.5,
                                print_hht_time=True,
                                print_emd_time=True)

#hilbert_spectrum, freqAxis = calc_hilbert_spectrum(input_signal1)

plot_hilbert_spectrum(hilbert_spectrum, freqAxis, fs, show=False)


# Fit the exponential decay model to the data
#damping_info = damping_analysis(hilbert_spectrum)


plt.show()