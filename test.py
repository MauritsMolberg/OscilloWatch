import numpy as np
from emd import emd
from hht import hht, calc_hilbert_spectrum
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

def interpolate_signal(input_signal):
    # Find the indices of non-None values
    non_none_indices = [i for i, value in enumerate(input_signal) if value is not None]

    # If there are no non-None values, return the original signal
    if not non_none_indices:
        return input_signal

    # Get the corresponding non-None values and their indices
    non_none_values = [input_signal[i] for i in non_none_indices]

    # Create an interpolation function using linear interpolation
    interpolation_function = interp1d(non_none_indices, non_none_values, kind='linear', fill_value='extrapolate')

    # Find the first index that is not None
    first_non_none_index = non_none_indices[0]

    # Generate the new signal with interpolated values
    new_signal = [interpolation_function(i) if value is None else value for i, value in enumerate(input_signal, start=first_non_none_index)]

    return new_signal


# Example usage:
input_signal = [None, None, 1, 3, None, 7, None, None, 10]
output_signal = interpolate_signal(input_signal)
print(output_signal)
"""


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