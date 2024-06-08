import numpy as np
import matplotlib.pyplot as plt

from OscilloWatch.SegmentAnalysis import SegmentAnalysis
from OscilloWatch.OWSettings import OWSettings


def f(t):
    return (
        20*np.exp(-.2*t)*np.cos(10*np.pi*t)
        + 8*np.exp(.1*t)*np.cos(5*np.pi*t)
        + 10*np.exp(.15*t)*np.cos(2.4*np.pi*t)
        + 16*np.exp(.1)*np.cos(np.pi*t)
    )


start = -2
end = 12
fs = 50
t = np.arange(start, end, 1/fs)
input_signal1 = f(t)


settings = OWSettings(
    extension_padding_time_start=2,
    extension_padding_time_end=2,
    print_emd_time=True,
    print_hht_time=True,
    minimum_frequency=0.1
)


plt.figure()
plt.plot(t, input_signal1)

# Comment out if you change the function or start/end.
plt.axis((-2.5, 12.5, -70.0, 120.0))
plt.axvline(x=0, ymin=0.2, ymax=.85, ls="--", color="red")
plt.axvline(x=10, ymin=0.2, ymax=.85, ls="--", color="red")

plt.xlabel("t [s]")
plt.ylabel("f(t)")
plt.tight_layout()


seg_an = SegmentAnalysis(input_signal1, settings)
seg_an.analyze_segment()
seg_an.hht.emd.plot_emd_results(show=False)
seg_an.hht.plot_hilbert_spectrum(show=False)


for i in range(len(seg_an.mode_info_list)):
    print(seg_an.mode_info_list[i], "\n")


plt.show()
