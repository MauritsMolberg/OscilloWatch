from OscilloWatch.RealTimeAnalysis import RealTimeAnalysis
from OscilloWatch.OWSettings import OWSettings

settings_N44 = OWSettings(segment_length_time=3,
                          extension_padding_time_start=.5,
                          extension_padding_time_end=0,
                          channel="bus_3000",
                          ip="10.100.0.75",
                          port=34702,
                          pmu_id=3000,
                          sender_device_id=45,
                          print_alarms=True
                          )

settings_real_NTNU = OWSettings(segment_length_time=10,
                                extension_padding_time_start=3,
                                extension_padding_time_end=3,
                                minimum_amplitude=0.1,
                                channel="VAYPM",
                                ip="129.241.31.57",
                                port=10111,
                                pmu_id=2017,
                                sender_device_id=2017,
                                results_file_path="../results/Real-time/real_NTNU",
                                print_alarms=True,
                                print_segment_analysis_time=True,
                                minimum_frequency=0.1
                                )

settings_zurich = OWSettings(segment_length_time=10,
                             extension_padding_time_start=3,
                             extension_padding_time_end=3,
                             channel="VA1",
                             ip="160.85.6.44",
                             port=4801,
                             pmu_id=200,
                             sender_device_id=4321,
                             results_file_path="../results/Real-time/Zurich",
                             print_alarms=True
                             )

settings_array = OWSettings(segment_length_time=10,
                            extension_padding_time_start=5,
                            extension_padding_time_end=2,
                            channel="signal",
                            pmu_id=1410,
                            print_alarms=True
                            )

settings_simplePMU = OWSettings(segment_length_time=10,
                                extension_padding_time_start=2,
                                extension_padding_time_end=.5,
                                channel="Phasor2.3",
                                pmu_id=1411,
                                print_alarms=True
                                )

settings_topsrt = OWSettings(segment_length_time=10,
                             extension_padding_time_start=2,
                             extension_padding_time_end=.5,
                             channel="V",
                             pmu_id=1,
                             sender_device_id=1,
                             print_alarms=True
                             )

rta = RealTimeAnalysis(settings_real_NTNU)
rta.start()
