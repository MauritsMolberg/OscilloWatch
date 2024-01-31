from synchrophasor.pmu_mod import Pmu
import socket


"""
tinyPMU will listen on ip:port for incoming connections.
When tinyPMU receives command to start sending
measurements - fixed (sample) measurement will
be sent.
"""


if __name__ == "__main__":

    ip = socket.gethostbyname(socket.gethostname())
    ip = "localhost"
    pmu = Pmu(ip=ip, port=50000)
    pmu.logger.setLevel("DEBUG")

    pmu.set_configuration()  # This will load default PMU configuration specified in IEEE C37.118.2 - Annex D (Table D.2)
    pmu.set_header()  # This will load default header message "Hello I'm tinyPMU!"

    pmu.run()  # PMU starts listening for incoming connections

    while True:
        if pmu.clients:  # Check if there is any connected PDCs
            pmu.send(pmu.ieee_data_sample)  # Sending sample data frame specified in IEEE C37.118.2 - Annex D (Table D.1)

    pmu.join()
