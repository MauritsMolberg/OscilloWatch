import matplotlib.pyplot as plt
import numpy as np
from emd import emd
from hht import hht, calc_hilbert_spectrum, split_signal_freq_change
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy.signal import stft, find_peaks




if __name__ == "__main__":
    a = np.array([0,0,0,0,0,1,1,1,1,1,0,0,0,-1,0])
    peaks, _ = find_peaks(np.abs(a))
    remaining = np.copy(a)
    print("Original:", remaining)
    for ind in peaks:
        print("Added to list:", remaining[:ind])
        remaining = remaining[ind:]
        peaks -= ind
        print(f"Peak at {ind}. Remaining: {remaining}")

"""
    fs = 50
    nperseg = 100

    time1 = np.arange(0, 4, 1/fs)  # 4 seconds
    signal1 = np.sin(2 * np.pi * 5 * time1)

    time2 = np.arange(4, 10, 1/fs)  # 6 seconds
    signal2 = np.sin(2 * np.pi * 4 * time2)

    # Concatenate the two signals
    joined_signal = np.concatenate([signal1, signal2])




    f,t,Zxx = stft(joined_signal, fs=fs, nperseg=nperseg)

    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    split_signal_freq_change(joined_signal, fs=fs, threshold=1, nperseg=nperseg)


    plt.show()

def interpolate_signal(input_signal):
    # Find the indices of non-None values
    non_none_indices = [i for i, value in enumerate(input_signal) if value is not None]

    # If there are no non-None values, return the original signal
    if not non_none_indices:
        return input_signal


    # Find the first index that is not None
    first_non_none_index = non_none_indices[0]

    # Ignore all None values before the first non-None value
    cropped_signal = input_signal[first_non_none_index:]

    # Recalculate non_none_indices based on the cropped signal
    non_none_indices = [i for i, value in enumerate(cropped_signal) if value is not None]


    # Get the corresponding non-None values and their indices
    non_none_values = [cropped_signal[i] for i in non_none_indices]

    # Create a cubic spline interpolation function
    spline = CubicSpline(non_none_indices, non_none_values)

    # Generate the new signal with interpolated values
    new_signal = [float(spline(i)) if value is None else value for i, value in enumerate(cropped_signal)]

    return new_signal


# Example usage:
input_signal = [None, None, 2, 4, None, None, None, 64, None, None, 512]
output_signal = interpolate_signal(input_signal)
print(output_signal)

plt.figure()
plt.plot([2,4,8,16,32,64,128,256,512])
plt.plot(output_signal)
plt.show()




# Damped sinusoidal model function
def damped_sinusoidal_model(t, A, omega, zeta, phi):
    return A * np.exp(-zeta * omega * t) * np.sin(omega * t + phi)

# Sample data for irregularly sampled time axis
time_points = np.linspace(0, 10, num=100)
amplitude = 2.0 * np.exp(-0.3 * time_points) * np.sin(2 * np.pi * 0.5 * time_points)

# Fit the model using np.linspace
popt_linspace, _ = curve_fit(damped_sinusoidal_model, time_points, amplitude)

# Fit the model using np.arange (assuming regularly sampled data)
popt_arange, _ = curve_fit(damped_sinusoidal_model, np.arange(len(amplitude)), amplitude)

print("Fitted Parameters using np.linspace:", popt_linspace)
print("Fitted Parameters using np.arange:", popt_arange)


"""