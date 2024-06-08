# OscilloWatch

---

PMU data monitoring tool that detects and gives early warnings for poorly
damped low-frequency oscillations in power systems.

Uses the Hilbert-Huang transform (HHT) on segments of data samples and uses an
algorithm to find and characterize oscillatory modes from the Hilbert spectrum.
Raises an alarm if a sustained with poor damping is found.

Made for my master's thesis, which describes how it works in great detail. It
will be linked here when published.

Please note that this is meant as a proof of concept and something
that can be built upon to become useful. It does not have a proper UI
and is unlikely to be very useful in an actual control room its current state.

---

## Basic Usage

---

### Installation

To install with pip:

    pip install OscilloWatch

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
users likely need to check/change, and advanced settings, which is
intended only for advanced users who know how the application works.

---

### Signal Analysis Modes

There are two options for analyzing a signal: Signal snapshot
analysis and real-time analysis. Below are basic explanations for
how to use them. See example scripts in the `examples` folder for
practical examples.

###

#### Signal Snapshot Analysis Mode
Analyzes a pre-given signal as if it were analyzed in real time.

1. Obtain signal in a list or array.
2. Create `OWSettings` object with the settings you want.
3. Create `SignalSnapshotAnalysis` object with your `OWSettings` object
and input signal as parameters.
4. Run analysis.


    # input_signal contains the signal that will be analyzed

    settings = OWSettings(
        segment_length_time=10,
        extension_padding_time_start=10,
        extension_padding_time_end=2,
        minimum_frequency=0.1
        minimum_amplitude=1e-4
    )

    sig_an = SignalSnapshotAnalysis(input_signal, settings)

    sig_an.analyze_whole_signal()

###

#### Real-Time Analysis Mode
Connects to a PMU or PDC and analyzes the data in real time.

1. Create `OWSettings` with the settings you want, including details
on the device you want to connect to.
2. Create `RealTimeAnalysis` object with your `OWSettings` object as
a parameter.
3. Run analysis.


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
        minimum_frequency=0.1
        minimum_amplitude=1e-4
    )

    rta = RealTimeAnalysis(settings)
    rta.start()  # Infinite loop

---

### Results
Numerical results are stored in table form in a CSV file. Explanations
for the results are given [here](results.md). They are  split into
basic and advanced results, similar to the settings. By default, only
the basic results are included by default. Advanced results can be
included by enabling the setting `include_advanced_results`.

Additionally, results are stored in a PKL file with the Pickle library.
This allows advanced users to access all variables, plots etc. from the
analysis, and it can be used for the post-processing methods in the
`examples/post_processing` folder.