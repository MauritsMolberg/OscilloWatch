from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np


def EMD(input_signal, max_iterations=10, zero_padding = False):
    imf_list = []
    xAxis = np.linspace(0, len(input_signal), len(input_signal))
    res = input_signal
    while len(imf_list) < max_iterations:
        upper_peaks, p = find_peaks(res)
        lower_peaks, p = find_peaks(-res)

        interp_upper = interp1d(upper_peaks, res[upper_peaks], kind = "cubic", fill_value="extrapolate")
        interp_lower = interp1d(lower_peaks, res[lower_peaks], kind = "cubic", fill_value="extrapolate")

        upper_envelope = interp_upper(xAxis)
        lower_envelope = interp_lower(xAxis)

        if zero_padding:
            upper_envelope[0:5] = 0
            upper_envelope[-5:] = 0
            lower_envelope[0:5] = 0
            lower_envelope[-5:] = 0

        res_old = res
        res = (upper_envelope + lower_envelope)/2
        imf = res_old - res

        imf_list.append(imf)

        plt.figure(figsize=(10, 8))
        plt.plot(xAxis, res_old, label='signal')
        plt.plot(xAxis, upper_envelope, label='upper envelope')
        plt.plot(xAxis, lower_envelope, label='lower envelope')
        plt.plot(xAxis, res, label='average envelope')
        plt.title('Visualizing envelopes iteration ' + str(len(imf_list)))
        plt.xlim(0, 100)
        plt.legend(loc='lower right')

    return res, imf_list

np.random.seed(0)
#input_signal = np.random.randn(500)

def f(t):
    return 10*np.exp(.2*t)*np.cos(2.4*np.pi*t) + 8*np.exp(-.1*t)*np.cos(np.pi*t)

t = np.linspace(0, 5, 1000)

plt.figure()
plt.plot(t, f(t))

input_signal = np.array([f(x) for x in range(len(t))])

res, imf_list = EMD(input_signal, max_iterations=2)


#xAxis = np.linspace(0, len(input_signal), len(input_signal))
#plt.figure()
#for i in range(len(imf_list)):
#    plt.subplot((len(imf_list)+1)//2 + 1, 2, i+1)
#    plt.plot(xAxis, imf_list[i])
#    plt.title("imf"+str(i+1))


#plt.subplot((len(imf_list)+1)//2 + 1, 2, len(imf_list)+1)
#plt.plot(xAxis, res)
#plt.title("res")

#plt.plot(xAxis, input_signal)
#plt.plot(xAxis, imf_list[0])

#plt.plot(xAxis, res, label="res")
#plt.plot(xAxis, sig)
#plt.plot(upper_peaks, sig[upper_peaks], "o")
#plt.plot(lower_peaks, sig[lower_peaks], "o")
#plt.plot(xAxis, upper_envelope)
#plt.plot(xAxis, lower_envelope)
#plt.plot(xAxis, res, label = "res")
#plt.plot(xAxis, imf, label = "imf")
#plt.xlabel("Sample")
#plt.legend()

plt.show()