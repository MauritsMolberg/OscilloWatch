"""
Sends values of a function as voltage magnitude samples in data frames as a PMU.
"""

import socket
import time

import numpy as np
from synchrophasor.frame import ConfigFrame2
from synchrophasor.simplePMU import SimplePMU


if __name__ == "__main__":
    ip = "localhost"
    port = 50000

    fs = 50

    def f(t):  # Function that will be sent
        return (
                10*np.exp(.15*t)*np.cos(2.4*np.pi*t)
                #+ 16*np.exp(.1)*np.cos(np.pi*t)
                + 8*np.exp(.1*t)*np.cos(5*np.pi*t)
                + 20*np.exp(-.2*t)*np.cos(10*np.pi*t)
                )

    station_names = ["FunctionPMU"]
    channel_names = [["signal"]]
    channel_types = [["v"]]
    id_codes = [1410]

    pmu = SimplePMU(
        ip, port,
        publish_frequency=fs,
        station_names=station_names,
        channel_names=channel_names,
        channel_types=channel_types,
        pdc_id=1,
        id_codes=id_codes
    )

    pmu.pmu.set_header("I'm a simulated PMU sending data from Python.")
    pmu.run()

    #array_copy = np.copy(array)

    k = 0
    while True:
        time.sleep(1/fs)
        if pmu.pmu.clients:
            pmu.publish(phasor_data=[(f(k/fs), 0)])
            #array_copy = np.delete(array_copy, 0)
            k += 1

    pmu.cleanup()
