import time

import numpy as np
from synchrophasor.pdc import Pdc
from synchrophasor.frame import DataFrame
import matplotlib.pyplot as plt

"""
tinyPDC will connect to pmu_ip:pmu_port and send request
for header message, configuration and eventually
to start sending measurements.
"""
import socket


if __name__ == "__main__":

    ip = socket.gethostbyname(socket.gethostname())
    #ip = "192.168.0.7"
    pdc = Pdc(pdc_id=1410, pmu_ip=ip, pmu_port=50000)
    pdc.logger.setLevel("DEBUG")

    pdc.run()  # Connect to PMU
    
    # pdc.stop()
    config = pdc.get_config()  # Get configuration from PMU
    header = pdc.get_header()

    pdc.start()  # Request to start sending measurements
   
    i = 0
    data_array = []
    while i < 100:
        data = pdc.get()  # Keep receiving data

        if type(data) == DataFrame:
            data_array.append(data.get_measurements())
        else:
            if not data:
                pdc.quit()  # Close connection
                break

            if len(data)>0:
                for meas in data: 
                    data_array.append(meas.get_measurements())

        i += 1

    phasor_array = np.array([])
    for data in data_array:
        # data["measurements"][pmu_index]["phasors"][phasor_index][0 for magnitude, 1 for angle]
        phasor = data["measurements"][0]["phasors"][0][0]
        phasor_array = np.append(phasor_array, phasor)

    plt.figure()
    plt.plot(phasor_array)
    plt.show()

