import numpy as np
import matplotlib.pyplot as plt

from OscilloWatch.OWSettings import OWSettings
from OscilloWatch.EMD import EMD
from OscilloWatch.HHT import HHT
from OscilloWatch.SignalSnapshotAnalysis import SignalSnapshotAnalysis


def f(t):
    return (
            20*np.exp(-.2*t)*np.cos(10*np.pi*t)
            + 8*np.exp(.1*t)*np.cos(5*np.pi*t)
            + 10*np.exp(.15*t)*np.cos(2.4*np.pi*t)
            + 16*np.exp(.1)*np.cos(np.pi*t)
    )


start = -12
end = 12
fs = 50

t = np.arange(start, end, 1/fs)
input_signal = f(t)

plt.figure()
plt.plot(t, input_signal)
plt.xlabel("t [s]")
plt.ylabel("f(t)")
plt.tight_layout()


settings = OWSettings(
    fs=fs,
    segment_length_time=10,
    extension_padding_time_start=2,
    extension_padding_time_end=2,
    csv_delimiter=",",  # Change to ";" if your Excel interprets "," as decimal separator
    print_emd_time=True,
    print_hht_time=True,
    include_advanced_results=True,
    minimum_frequency=0.1,
    results_file_path="results/synth_signal"
)

sig_an = SignalSnapshotAnalysis(input_signal, settings)
sig_an.analyze_whole_signal()

for i, segment in enumerate(sig_an.segment_analysis_list):
    print(f"Segment {i}:\n{segment.mode_info_list}\n\n")
    segment.hht.emd.plot_emd_results(show=False)
    segment.hht.plot_hilbert_spectrum(show=False)

plt.show()
