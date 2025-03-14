# OscilloWatch

---

PMU data monitoring tool that detects poorly damped oscillations in power systems
in real time. Made for my [master's thesis](https://hdl.handle.net/11250/3147677).

Divides a signal into segments of data samples, which are analyzed using the
Hilbert-Huang transform (HHT) and an oscillation detection algorithm, to detect
and characterize oscillatory modes from the Hilbert spectrum. Raises alarms for
sustained oscillations with poor damping that are detected.

Please note that this is meant as a proof of concept and something
that can be built upon to become a useful tool for grid operators. It does not have
a proper user interface and is not ready for deployment in an actual control room
its current state.

---

## Basic Usage

---

### Installation

To install with pip:

    pip install OscilloWatch

    pip install synchrophasor @ git+https://github.com/hallvar-h/pypmu

(Synchrophasor (pyPMU), which is required for real-time analysis, must
be installed manually because it uses a specific fork only accessible
through direct GitHub link, which PyPI does not allow.)

You can also install it by downloading the zip file with the source
code or by cloning with Git. The following dependencies must then be
installed manually:

* numpy
* scipy
* matplotlib
* pyPMU ([this fork](https://github.com/hallvar-h/pypmu))

---

### Settings

Settings are assigned by creating an `OWSettings` object with the
desired settings as constructor parameters. All settings are explained
[here](settings.md). They are split into basic settings, which most
users likely need to check/change, and advanced settings, which are
intended for advanced users who know how the application works.

---

### Signal Analysis Modes

There are two options for analyzing a signal: Signal snapshot
analysis and real-time analysis. Below are basic explanations for
how to use them, with basic examples intended as introductions. To learn
how to use the application, I recommend starting with the example scripts
in the `examples` folder and modifying them to fit your needs.

###

#### Signal Snapshot Analysis Mode
Analyzes a pre-given signal (such as a synthetic signal or pre-recorded PMU
data) as if it were analyzed in real time.

1. Obtain signal in a list or array.
2. Create `OWSettings` object with the settings you want.
3. Create `SignalSnapshotAnalysis` object with your `OWSettings` object
and input signal as parameters.
4. Run analysis.

```python
from OscilloWatch.OWSettings import OWSettings
from OscilloWatch.SignalSnapshotAnalysis import SignalSnapshotAnalysis

# input_signal is a list or array with the signal that will be analyzed.

settings = OWSettings(
    segment_length_time=10,
    extension_padding_time_start=10,
    extension_padding_time_end=2,
    minimum_frequency=0.1,
    minimum_amplitude=1e-4
)

sig_an = SignalSnapshotAnalysis(input_signal, settings)

sig_an.analyze_whole_signal()
```

###

#### Real-Time Analysis Mode
Connects to a PMU or PDC and analyzes the data in real time.

1. Create `OWSettings` with the settings you want, including details
on the device you want to connect to.
2. Create `RealTimeAnalysis` object with your `OWSettings` object as
a parameter.
3. Run analysis.

```python
from OscilloWatch.OWSettings import OWSettings
from OscilloWatch.RealTimeAnalysis import RealTimeAnalysis

settings = OWSettings(
    ip="192.168.1.10",
    port=50000,
    sender_device_id=1410,
    pmu_id=1410,
    channel="VA",
    phasor_component="magnitude",

    segment_length_time=10,
    extension_padding_time_start=10,
    extension_padding_time_end=2,
    minimum_frequency=0.1,
    minimum_amplitude=1e-4
)

rta = RealTimeAnalysis(settings)
rta.start()  # Infinite loop
```
---

### Results
Numerical results are stored in table form in a CSV file. Explanations
for the results are given [here](results.md). They are split into
basic and advanced results, similar to the settings. By default, only
the basic results are included by default. Advanced results can be
included by enabling the setting `include_advanced_results`.

Additionally, results are stored in a PKL file with the Pickle library.
This allows advanced users to access all variables, plots etc. from the
analysis, and it can be used for the post-processing methods in the
`examples/post_processing` folder.
