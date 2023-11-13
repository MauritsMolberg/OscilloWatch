from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np
from time import time


def get_envelopes(signal, interp_method="cubic"):
    upper_peaks, _ = find_peaks(signal)
    lower_peaks, _ = find_peaks(-signal)

    if len(upper_peaks) < 4 or len(lower_peaks) < 4:  #Unable to extract more IMFs
        return None, None

    xAxis = np.arange(len(signal))
    interp_upper = CubicSpline(upper_peaks, signal[upper_peaks])
    interp_lower = CubicSpline(lower_peaks, signal[lower_peaks])

    upper_envelope = interp_upper(xAxis)
    lower_envelope = interp_lower(xAxis)

    return upper_envelope, lower_envelope


def emd(input_signal,
        sd_tolerance=.2,
        max_imfs=10,
        max_sifting_iterations = 30,
        mirror_padding_fraction = .5,
        remove_padding = False,
        print_sifting_details = False,
        print_time = False):

    start_time = time()

    imf_list = []
    r = np.copy(input_signal) # Updated every time an IMF is subtracted from it
    original_length = len(input_signal)

    # Mirror padding, to handle boundary effects
    if mirror_padding_fraction > 0: # Skip if zero or negative
        num_samples_to_mirror = int(original_length*mirror_padding_fraction)

        mirrored_fraction_start = r[:num_samples_to_mirror][::-1]
        mirrored_fraction_end = r[-num_samples_to_mirror:][::-1]
        r = np.concatenate((mirrored_fraction_start, r, mirrored_fraction_end))


    # Set to True if no more IMFs can be extracted, i.e. if residual has less than 4 upper and lower peaks.
    # Breaks out of both loops if True.
    emd_done = False

    for n in range(max_imfs):
        h = np.copy(r)

        if print_sifting_details:
            print("----------------\nIMF iteration:", n+1)

        for k in range(max_sifting_iterations):
            upper_envelope, lower_envelope = get_envelopes(h)

            if upper_envelope is None or lower_envelope is None:  # h has less than 4 upper and lower peaks
                emd_done = True  # Breaks out of both loops
                break

            avg_envelope = (upper_envelope + lower_envelope) / 2
            h_old = np.copy(h)
            h -= avg_envelope

            if print_sifting_details:
                print("Sifting iteration:", k+1, "\nSD:", np.std(h-h_old, dtype=np.float64))

            if np.std(h-h_old, dtype=np.float64) < sd_tolerance:
                break

        if emd_done:
            break

        imf_list.append(h)
        r -= h


    # Removing the extensions added for padding if desired
    if remove_padding:
        ext_len = int(original_length * mirror_padding_fraction)
        imf_list = [imf[ext_len:-ext_len] for imf in imf_list]
        r = r[ext_len:-ext_len]

    if print_time:
        print(f"EMD completed in {time()-start_time:.3f} seconds.")

    return imf_list, r


def plot_emd_results(input_signal, imf_list, residual, sampling_freq, show = True):
    num_imfs = len(imf_list)

    # Need different time axes for the input signal and IMFs in case the IMFs have padding that is not removed.
    tAxis = np.linspace(0, len(input_signal)/sampling_freq, len(input_signal))
    tAxis_imf = np.linspace(0, len(imf_list[0])/sampling_freq, len(imf_list[0]))

    # Create a grid of subplots for IMFs and residual
    fig, axes = plt.subplots(num_imfs + 2, 1, figsize=(8, 2 * (num_imfs + 1)), sharex=True)

    # Plot the input signal
    axes[0].plot(tAxis, input_signal, color='blue', linewidth = .7)
    axes[0].set_title('Input Signal')

    # Plot each IMF
    for i, imf in enumerate(imf_list):
        axes[i + 1].plot(tAxis_imf, imf, color='green', linewidth = .7)
        axes[i + 1].set_title(f'IMF {i + 1}')

    # Plot the residual
    axes[num_imfs+1].plot(tAxis_imf, residual, color='red', linewidth = .7)
    axes[num_imfs+1].set_title('Residual')

    # Set labels
    for ax in axes:
        ax.set_ylabel('Amplitude')
    axes[num_imfs+1].set_xlabel('Time [s]')

    plt.tight_layout()
    if show:
        plt.show()


"""
Plot peaks:
plt.plot(upper_peaks/fs,x[upper_peaks],'o',label = 'Upper peaks')
plt.plot(lower_peaks/fs,x[lower_peaks],'o',label = 'Lower peaks')
"""