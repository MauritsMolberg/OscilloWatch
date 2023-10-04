import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def get_envelopes(signal, interp_method="cubic"):
    upper_peaks, _ = find_peaks(signal)
    lower_peaks, _ = find_peaks(-signal)

    if len(upper_peaks) < 4 or len(lower_peaks) < 4:
        return None, None

    xAxis = np.arange(len(signal))
    interp_upper = interp1d(upper_peaks, signal[upper_peaks], kind=interp_method, fill_value="extrapolate")
    interp_lower = interp1d(lower_peaks, signal[lower_peaks], kind=interp_method, fill_value="extrapolate")

    upper_envelope = interp_upper(xAxis)
    lower_envelope = interp_lower(xAxis)

    return upper_envelope, lower_envelope

def emd(input_signal, max_imfs=10, max_iterations=, sd_threshold=2, zero_padding=False):
    imf_list = []
    sifted_signal = np.copy(input_signal)

    for _ in range(max_imfs):
        res = np.copy(sifted_signal)
        for _ in range(max_iterations):
            upper_envelope, lower_envelope = get_envelopes(res)

            if upper_envelope is None or lower_envelope is None:
                break

            if zero_padding:
                upper_envelope[0:50] = 0
                upper_envelope[-50:] = 0
                lower_envelope[0:50] = 0
                lower_envelope[-50:] = 0

            avg_envelope = (upper_envelope + lower_envelope) / 2
            res_old = np.copy(res)
            res = res - avg_envelope

            print(np.std(res - res_old))
            if np.std(res - res_old) < sd_threshold:
                print("--------")
                break

        imf_list.append(res)
        sifted_signal -= res

    return sifted_signal, imf_list


def plot_emd_results(input_signal, imf_list, residual):
    num_imfs = len(imf_list)

    # Create a grid of subplots for IMFs and residual
    fig, axes = plt.subplots(num_imfs + 2, 1, figsize=(8, 2 * (num_imfs + 1)))

    # Plot the input signal
    axes[0].plot(input_signal, label='Input Signal', color='blue')
    axes[0].set_title('Input Signal')

    # Plot each IMF
    for i, imf in enumerate(imf_list):
        axes[i + 1].plot(imf, label=f'IMF {i + 1}', color='green')
        axes[i + 1].set_title(f'IMF {i + 1}')

    # Plot the residual
    axes[num_imfs+1].plot(residual, label='Residual', color='red')
    axes[num_imfs+1].set_title('Residual')

    # Set labels and legend for all subplots
    for ax in axes:
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.legend()

    plt.tight_layout()
    plt.show()



def f(t):
    return 10*np.exp(.2*t)*np.cos(2.4*np.pi*t) + 8*np.exp(-.1*t)*np.cos(np.pi*t)


start = -10
end = 15
fs = 100


input_signal = f(np.arange(start, end, 1/fs))


res, imf_list = emd(input_signal, max_imfs=3, zero_padding=True)


plot_emd_results(input_signal, imf_list, res)


#plt.figure()
#for i in range(len(imf_list)):
#    plt.subplot((len(imf_list)+1)//2 + 1, 2, i+1)
#    plt.plot(xAxis, imf_list[i])
#    plt.title("imf"+str(i+1))


#plt.subplot((len(imf_list)+1)//2 + 1, 2, len(imf_list)+1)
#plt.plot(xAxis, res)
#plt.title("res")


#plt.figure()
#plt.plot(xAxis, input_signal)
#plt.title("Original signal")
#plt.show()