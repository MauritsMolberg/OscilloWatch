from OscilloWatch.post_processing import summarize_alarms

summarize_alarms([
                  ("../results/results_1.pkl", "PMU1"),
                  ("../results/results_2.pkl", "PMU2")
                 ], csv_delimiter=";")
