import socket

import numpy as np
from synchrophasor.frame import ConfigFrame2
from synchrophasor.pmu_mod import Pmu

def send_array(array, ip="localhost", port=50000):
    """
    Sends individual values from an array one at a time in real-time as a PMU, using the IEEE C37.118 protocol.
    """

    pmu = Pmu(ip=ip, port=port)
    pmu.logger.setLevel("DEBUG")

    cfg = ConfigFrame2(1410,  # PMU_ID
                       1000000,  # TIME_BASE
                       1,  # Number of PMUs included in data frame
                       "Random Station",  # Station name
                       1410,  # Data-stream ID(s)
                       (True, True, True, True),  # Data format - POLAR; PH - REAL; AN - REAL; FREQ - REAL;
                       1,  # Number of phasors
                       1,  # Number of analog values
                       1,  # Number of digital status words
                       ["signal", "ANALOG1", "BREAKER 1 STATUS",
                        "BREAKER 2 STATUS", "BREAKER 3 STATUS", "BREAKER 4 STATUS", "BREAKER 5 STATUS",
                        "BREAKER 6 STATUS", "BREAKER 7 STATUS", "BREAKER 8 STATUS", "BREAKER 9 STATUS",
                        "BREAKER A STATUS", "BREAKER B STATUS", "BREAKER C STATUS", "BREAKER D STATUS",
                        "BREAKER E STATUS", "BREAKER F STATUS", "BREAKER G STATUS"],  # Channel Names
                       [(0, "v")],  # Conversion factor for phasor channels - (float representation, not important)
                       [(1, "pow")],  # Conversion factor for analog channels
                       [(0x0000, 0xffff)],  # Mask words for digital status words
                       50,  # Nominal frequency
                       1,  # Configuration change count
                       50)  # Rate of phasor data transmission (this one is probably wrong, but irrelevant for testing)

    pmu.set_configuration(cfg)
    pmu.set_header("I'm a PMU sending data from an array, using the send_array function.")

    pmu.run()

    array_copy = np.copy(array)

    while len(array_copy):
        if pmu.clients:
            pmu.send_data(phasors=[(array_copy[0], 0)],
                          analog=[9.91],
                          digital=[0x0001])
            array_copy = np.delete(array_copy, 0)

    pmu.join()


if __name__ == "__main__":
    start = 0
    end = 100
    fs = 50
    t = np.arange(start, end, 1/fs)

    def f(t):
        return 3*np.sin(4*np.pi*t)

    input_signal1 = f(t)

    send_array(input_signal1)