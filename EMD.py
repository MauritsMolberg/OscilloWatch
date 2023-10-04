from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np


def get_envelopes(signal, interp_method="cubic"):
    upper_peaks, p = find_peaks(signal)
    lower_peaks, p = find_peaks(-signal)

    if len(upper_peaks) < 4 or len(lower_peaks) < 4:  #Unable to extract more IMFs
        return None, None

    xAxis = np.arange(len(signal))
    interp_upper = interp1d(upper_peaks, signal[upper_peaks], kind=interp_method, fill_value="extrapolate")
    interp_lower = interp1d(lower_peaks, signal[lower_peaks], kind=interp_method, fill_value="extrapolate")

    upper_envelope = interp_upper(xAxis)
    lower_envelope = interp_lower(xAxis)

    return upper_envelope, lower_envelope


def EMD(input_signal, sd_threshold=.1, max_imfs=10, max_iterations = 10, mirror_padding_fraction = .5):
    imf_list = []
    sifted_signal = np.copy(input_signal) #r0 = y. Updated every time an IMF is subtracted from it
    original_length = len(input_signal)

    #Mirror padding, to handle boundary effects
    if mirror_padding_fraction > 0:
        num_samples_to_mirror = int(original_length*mirror_padding_fraction)

        mirrored_fraction = sifted_signal[:num_samples_to_mirror][::-1]
        sifted_signal = np.concatenate((mirrored_fraction, sifted_signal, mirrored_fraction))


    for n in range(max_imfs):
        res = np.copy(sifted_signal) #h_k-1 = r_n-1
        for k in range(max_iterations):
            upper_envelope, lower_envelope = get_envelopes(res)

            if upper_envelope is None or lower_envelope is None:
                break

            avg_envelope = (upper_envelope + lower_envelope) / 2
            res_old = np.copy(res)
            res -= avg_envelope #h_k = h_k-1 - m_k-1

            print(np.std(res-res_old))

            if np.std(res-res_old) < sd_threshold:
                break

        imf_list.append(res)
        sifted_signal -= res
        print("--------")


    #Remove the extensions added for padding
    if mirror_padding_fraction > 0:
        ext_len = int(original_length*mirror_padding_fraction)
        imf_list = [imf[ext_len:-ext_len] for imf in imf_list]
        sifted_signal = sifted_signal[ext_len:-ext_len]

    return sifted_signal, imf_list


def plot_emd_results(input_signal, imf_list, residual):
    num_imfs = len(imf_list)

    # Create a grid of subplots for IMFs and residual
    fig, axes = plt.subplots(num_imfs + 2, 1, figsize=(8, 2 * (num_imfs + 1)))

    # Plot the input signal
    axes[0].plot(input_signal, color='blue', linewidth = .7)
    axes[0].set_title('Input Signal')

    # Plot each IMF
    for i, imf in enumerate(imf_list):
        axes[i + 1].plot(imf, color='green', linewidth = .7)
        axes[i + 1].set_title(f'IMF {i + 1}')

    # Plot the residual
    axes[num_imfs+1].plot(residual, color='red', linewidth = .7)
    axes[num_imfs+1].set_title('Residual')

    # Set labels and legend for all subplots
    for ax in axes:
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')

    plt.tight_layout()
    #plt.show()

#input_signal = np.random.randn(500)

def f(t):
    return 10*np.exp(.2*t)*np.cos(2.4*np.pi*t) + 8*np.exp(-.1*t)*np.cos(np.pi*t)


start = -5
end = 10
fs = 100


#input_signal = f(np.arange(start, end, 1/fs))

np.random.seed(0)
input_signal = np.random.randn(500)

#f = open("array_print.txt", "w+")
#f.write("[")
#for element in input_signal:
#    f.write(str(element) + " ")
#f.write("]")
#f.close()


res, imf_list = EMD(input_signal, max_imfs=3)
plot_emd_results(input_signal, imf_list, res)



plt.show()