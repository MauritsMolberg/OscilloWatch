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


def emd(input_signal, sd_tolerance=.2, max_imfs=10, max_sifting_iterations = 30, mirror_padding_fraction = .5, print_sifting_details = False):
    imf_list = []
    r = np.copy(input_signal) # Updated every time an IMF is subtracted from it
    original_length = len(input_signal)

    # Mirror padding, to handle boundary effects
    if mirror_padding_fraction > 0: # Skip if zero or negative
        num_samples_to_mirror = int(original_length*mirror_padding_fraction)

        mirrored_fraction = r[:num_samples_to_mirror][::-1]
        r = np.concatenate((mirrored_fraction, r, mirrored_fraction))

    emd_done = False # If no more IMFs can be extracted, i.e. if residual has less than 4 upper and lower peaks. Breaks out of both loops if True.
    for n in range(max_imfs):
        h = np.copy(r) # First iteration: Applied to original signal. Otherwise: Previous IMF subtracted from signal.

        if print_sifting_details:
            print("----------------\nIMF iteration:", n+1)

        for k in range(max_sifting_iterations):
            upper_envelope, lower_envelope = get_envelopes(h)

            if upper_envelope is None or lower_envelope is None: # h has less than 4 upper and lower peaks
                emd_done = True # Breaks out of both loops
                break

            avg_envelope = (upper_envelope + lower_envelope) / 2
            h_old = np.copy(h)
            h -= avg_envelope

            if print_sifting_details:
                print("Sifting iteration:", k+1, "\nSD:", np.std(h-h_old))

            if np.std(h-h_old) < sd_tolerance:
                break

        if emd_done:
            break

        imf_list.append(h)
        r -= h


    #Remove the extensions added for padding
    if mirror_padding_fraction > 0:
        ext_len = int(original_length*mirror_padding_fraction)
        imf_list = [imf[ext_len:-ext_len] for imf in imf_list]
        r = r[ext_len:-ext_len]

    return imf_list, r


def plot_emd_results(input_signal, imf_list, residual, show = True):
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
    if show:
        plt.show()

