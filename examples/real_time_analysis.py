from methods.RealTimeAnalysis import RealTimeAnalysis
from methods.AnalysisSettings import AnalysisSettings

settings_N44 = AnalysisSettings(segment_length_time=3,
                                extension_padding_time_start=.5,
                                extension_padding_time_end=0,
                                channel="bus_3000",
                                ip="10.100.0.75",
                                port=34702,
                                pmu_id=3000,
                                sender_device_id=45
                                )

settings_array = AnalysisSettings(segment_length_time=10,
                                  extension_padding_time_start=0,
                                  extension_padding_time_end=0,
                                  channel="signal",
                                  pmu_id=1410
                                  )

settings_simplePMU = AnalysisSettings(segment_length_time=10,
                                      extension_padding_time_start=2,
                                      extension_padding_time_end=.5,
                                      channel="Phasor2.3",
                                      pmu_id=1411
                                      )

rta = RealTimeAnalysis(settings_array)
rta.run_analysis()
