import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from emd import emd


def f(t):
    return 10*np.exp(.2*t)*np.cos(2.4*np.pi*t) + 8*np.exp(-.1*t)*np.cos(np.pi*t)

start = 0
end = 5
fs = 100
input_signal1 = f(np.arange(start, end, 1/fs))

#Random (reproducable) signal
np.random.seed(0)
input_signal2 = np.random.randn(500)


#res1, imf_list1 = emd(input_signal1, max_imfs=4, mirror_padding_fraction=1)
#res2, imf_list2 = emd(input_signal2)

input_signal = np.copy(input_signal2)
res, imf_list = emd(input_signal, max_imfs=3, mirror_padding_fraction=1)
imf_sum = np.zeros(len(input_signal))

#plot_emd_results(input_signal, imf_list, res)

for i in range(len(imf_list)):
    imf_sum += imf_list[i]

plt.figure()
plt.plot(input_signal)

#plt.figure()
#plt.plot(imf_list[0])

plt.figure()
plt.plot(imf_sum)

plt.show()