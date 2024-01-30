import numpy as np
from synchrophasor.timeSeriesPlayback import PMUTimeSeriesPublisher
import threading
import time
import matplotlib.pyplot as plt
import socket

"""
tinyPMU will listen on ip:port for incoming connections.
When tinyPMU receives command to start sending
measurements - fixed (sample) measurement will
be sent.
"""


if __name__ == "__main__":
    # __file__ = r'/examples/timeSeriesPlayback.py'
    test = 2
    if test == 1:
        t_end = 10
        dt = 0.02

        # List of PMUs
        station_names = ['PMU1', 'PMU2', 'PMU3']

        # List of list of channels pr. PMU
        channel_names = [
            ['Phasor1.1', 'Phasor1.2'],
            ['Phasor2.1', 'Phasor2.2', 'Phasor2.3'],
            ['Phasor3.1'],
        ]

    else:  # test == 2
        t_end = 10
        dt = 0.02
        n_stations = 40
        n_channels = 1
        station_names = ['PMU{}'.format(i) for i in range(n_stations)]
        channel_names = [['Ph{}'.format(j) for j in range(n_channels)] for _ in range(n_stations)]

    # Generate some random time series
    t = np.arange(0, t_end, dt)
    phasor_data = []
    for channel_names_ in channel_names:
        angles = 1e-2*np.cumsum(np.random.randn(len(t), len(channel_names_)), axis=0)
        magnitudes = 1 + 1e-3*np.cumsum(np.random.randn(len(t), len(channel_names_)), axis=0)
        phasors_pmu = magnitudes*np.exp(1j*angles)
        phasors_pmu -= (phasors_pmu[-1, :] - phasors_pmu[0, :])[None, :] * t[:, None] / (t[-1] - t[0])

        phasor_data.append(phasors_pmu)

    # Plot the time series that are streamed
    fig, ax = plt.subplots(2, len(station_names), squeeze=False, sharex=True, sharey='row')
    for i, (ax_, phasors_pmu) in enumerate(zip(ax.T, phasor_data)):
        ax[0, i].set_title(station_names[i])
        ax[0, i].plot(t, abs(phasors_pmu))
        ax[1, i].plot(t, np.angle(phasors_pmu))
        ax[1, i].set_xlabel('Time [s]')
        ax[1, i].legend(channel_names[i])

    ax[0, 0].set_ylabel('Phasor magnitudes []')
    ax[1, 0].set_ylabel('Phasor angles [rad]')

    # PMU Snapshot Publisher
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    port = 50001

    pmu_publisher = PMUTimeSeriesPublisher(
        ip, port,
        time=t,
        # publish_frequency=10,  # is determined automatically if not specified
        phasors=phasor_data,
        station_names=station_names,
        channel_names=channel_names
    )
    pmu_publisher.pmu.run()  # Start listening to incoming connections and send data (if published)

    # Run the PMU Publisher in a separate thread to be able to exit
    pmu_thread = threading.Thread(target=pmu_publisher.main_loop, name='PMU-Publisher-Thread')
    pmu_thread.daemon = True
    pmu_thread.start()

    # Continues until the plot window is closed
    plt.show(block=True)
    pmu_publisher.stop()

    # This does not always work as intended, some threads/processes are not always stopped.
    pmu_publisher.cleanup()

    import threading

    for i, thread in enumerate(threading.enumerate()):
        print(thread)

