import random

from synchrophasor.frame import ConfigFrame2
from synchrophasor.pmu_mod import Pmu
import numpy as np
import time


"""
randomPMU will listen on ip:port for incoming connections.
After request to start sending measurements - random
values for phasors will be sent.
"""


if __name__ == "__main__":

    pmu = Pmu(ip="10.219.106.118", port=50000)
    pmu.logger.setLevel("DEBUG")

    stations = "Random Station"
    phasors = ["VA", "VB", "VC"]
    quantity = ['v', '']
    
    cfg = ConfigFrame2(1410,  # PMU_ID
                       1000000,  # TIME_BASE
                       1,  # Number of PMUs included in data frame
                       stations,  # Station name
                       1410,  # Data-stream ID(s)
                       (True, True, True, True),  # Data format - POLAR; PH - REAL; AN - REAL; FREQ - REAL;
                       len(phasors),  # Number of phasors
                       0,  # Number of analog values
                       0,  # Number of digital status words
                       [
                            *phasors,
                            #"ANALOG1",
                            #"BREAKER 1 STATUS",
                            #"BREAKER 2 STATUS", "BREAKER 3 STATUS", "BREAKER 4 STATUS", "BREAKER 5 STATUS",
                            #"BREAKER 6 STATUS", "BREAKER 7 STATUS", "BREAKER 8 STATUS", "BREAKER 9 STATUS",
                            #"BREAKER A STATUS", "BREAKER B STATUS", "BREAKER C STATUS", "BREAKER D STATUS",
                            #"BREAKER E STATUS", "BREAKER F STATUS", "BREAKER G STATUS"
                            ],  # Channel Names
                       [(0, "v"), (0, "i"),
                        (0, "v")],  # Conversion factor for phasor channels - (float representation, not important)
                       [],  # [(1, "pow")],  # Conversion factor for analog channels
                       [], # [(0x0000, 0xffff)],  # Mask words for digital status words
                       50,  # Nominal frequency
                       1,  # Configuration change count
                       30)  # Rate of phasor data transmission)

    pmu.set_configuration(cfg)
    pmu.set_header("Hey! I'm randomPMU! Guess what? I'm sending random measurements values!")

    pmu.run()

    while True:
        if pmu.clients:
            pmu.send_data(
                phasors=[
                    (random.uniform(215.0, 240.0), random.uniform(-0.1, 0.3)),
                    (random.uniform(215.0, 240.0), random.uniform(1.9, 2.2)),
                    (random.uniform(215.0, 240.0), random.uniform(3.0, 3.14))
                ],
                freq=50 + np.sin(time.time()),
                #analog=[9.91],
                #digital=[0x0001]
            )

    pmu.join()
