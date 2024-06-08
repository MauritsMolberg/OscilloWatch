## Basic results

* `Segment` Segment number, counting from 0
* `Timestamp` Timestamp from the data frame containing the first sample in the segment, not including padding samples. Seconds into the signal in snapshot analysis mode, date and time in real-time mode.
* `Mode status` "New" for modes that were not detected in the previous segment, and "Sustained" for modes that were detected in the previous segment.
* `Damping evaluation` Text field, alerting the user if the damping ratio is good, low, very low or negative.
* `Alarm` "Weak", "Strong" or "Critical" if alarm conditions are met, and the damping is low, very low or negative respectively. Blank if there is no alarm.
* `Frequency` Estimated frequency of the mode in hertz.
* `Damping ratio` Estimated damping ratio of the mode in the current segment.
* `Median amplitude` Median of the registered amplitudes in the amplitude curve.
* `Start time` The number of seconds into the segment when the oscillation is first detected.
* `End time` The number of seconds into the segment when the oscillation is last detected.

---

## Advanced results

* `Initial amplitude` The first non-zero value that is not removed for being part of a short consecutive sequence in the row.
* `Final amplitude` The lase non-zero value that is not removed for being part of a too short consecutive sequence in the row.
* `Frequency band start` The lowest frequency in the combined row in which the oscillation is detected.
* `Frequency band stop` The highest frequency in the combined row in which the oscillation is detected.
* `Initial amplitude estimate` The initial amplitude in the fitted curve. $A$ in $A e^{-\alpha t}$.
* `Decay rate` Decay rate of the fitted curve. $\alpha$ in $A e^{-\alpha t}$.
* `Non zero fraction (NZF)` The fraction of the whole (combined) row where the amplitude value is not zero. Includes zeros before start time and after end time.
* `Interpolated fraction` The fraction of the amplitude curve that was interpolated for being zero in the Hilbert spectrum. Does not include zeros before start time and after end time.
* `Coefficient of variation (CV)` The coefficient of variation between the amplitude curve (with interpolated gaps) and the fitted curve.