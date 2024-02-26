import time
from synchrophasor.simplePMU import SimplePMU
import socket


if __name__ == '__main__':

    publish_frequency = 5

    #ip = socket.gethostbyname(socket.gethostname())
    ip = "localhost"
    port = 50000

    station_names = ['PMU1', 'PMU2', 'PMU3']
    channel_names = [
        ['Phasor1.1', 'Phasor1.2'],
        ['Phasor2.1', 'Phasor2.2', 'Phasor2.3'],
        ['Phasor3.1'],
    ]
    channel_types = [
        ['v', 'i'],
        ['v', 'i', 'v'],
        ['i'],
    ]
    id_codes = [1410, 1411, 1412]

    # station_names = ['PMU1']
    # channel_names = [['Phasor1.1', 'Phasor1.2']]
    pmu = SimplePMU(
        ip, port,
        publish_frequency=publish_frequency,
        station_names=station_names,
        channel_names=channel_names,
        channel_types=channel_types,
        pdc_id=1,
        id_codes=id_codes
    )
    pmu.pmu.set_header("Test header")
    pmu.run()

    k = 0
    while True:
        time.sleep(1/publish_frequency)
        if pmu.pmu.clients:  # Check if there is any connected PDCs
            k += 1
            print(k)
            # t, v = input_signal

            # Publish C37.118-snapshot
            # pmu_data = [(mag, ang) for mag, ang in zip(np.abs(v), np.angle(v))]
            pmu.publish()

    pmu.cleanup()
    # # sys.exit()
    # import threading
    # for thread in threading.enumerate():
    #     print(thread.name)