from OscilloWatch.RealTimeAnalysis import RealTimeAnalysis
from OscilloWatch.OWSettings import OWSettings

# Receive from the send_function PMU script
settings_function = OWSettings(
    segment_length_time=10,
    extension_padding_time_start=5,
    extension_padding_time_end=2,
    csv_delimiter=",",  # Change to ";" if your Excel interprets "," as decimal separator
    channel="signal",
    pmu_id=1410,
    print_alarms=True,
    results_file_path="results/rt_function"
)

# Receive from the simplePMU PMU script
settings_simplePMU = OWSettings(
    segment_length_time=10,
    extension_padding_time_start=5,
    extension_padding_time_end=2,
    csv_delimiter=",",  # Change to ";" if your Excel interprets "," as decimal separator
    channel="Phasor2.3",
    pmu_id=1411,
    print_alarms=True,
    results_file_path="results/rt_simplePMU"
)


rta = RealTimeAnalysis(settings_function)
rta.start()
