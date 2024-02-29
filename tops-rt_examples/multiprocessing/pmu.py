import multiprocessing as mp
from topsrt.interfacing import QueueManager, InterfaceListener
from PySide6 import QtWidgets
from topsrt.rtsim_plot import SyncPlot
import sys
from topsrt.rtsim_plot import RTSimPlot
import tops.dynamic as dps
from topsrt.sim import RealTimeSimulatorThread
from topsrt.gui import LineOutageWidget
from topsrt.pmu import PMUPublisher


def main_pmu(qm_kwargs):
    import socket
    #ip = socket.gethostbyname(socket.gethostname())  # Get local ip automatically
    ip = "localhost"
    port = 50000

    manager = QueueManager(**qm_kwargs)
    manager.connect()

    app = QtWidgets.QApplication(sys.argv)
    # interface = RTSimPlot(n_samples=1000)
    # InterfaceListener.send_interface_init(manager, interface)

    sync_plot = SyncPlot(n_samples=1000, update_freq=50)
    tw_plot = RTSimPlot(n_samples=1000)
    pmus = PMUPublisher(publish_frequency=25, phasors=['v_g'], ip=ip, port=port)

    [InterfaceListener.send_interface_init(manager, interface) for interface in [sync_plot, pmus, tw_plot]]
    sync_plot.start()
    pmus.start()
    tw_plot.start()

    app.exec()

    return app


def main(qm_kwargs):

    manager = QueueManager(**qm_kwargs)
    manager.connect()
    init_queue = manager.get_init_queue()
    interface_listener = InterfaceListener(init_queue)

    # import tops.ps_models.k2a as model_data
    # import examples.test_systems.n44_ctrl as model_data
    import tops.ps_models.ieee39 as model_data

    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    ps.ode_fun(0, ps.x0)
    rts = RealTimeSimulatorThread(ps, dt=10e-3, speed=1)

    interface_listener.connect(rts)
    interface_listener.start()

    app = QtWidgets.QApplication(sys.argv)

    # Add Control Widgets
    line_outage_ctrl = LineOutageWidget(rts)

    rts.start()
    app.exec()
    rts.stop()

    return app


def main_server(qm_kwargs):
    manager = QueueManager(server=True, **qm_kwargs)
    manager.start()


if __name__ == '__main__':

    import socket
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    ip = "localhost"
    qm_kwargs = dict(address=(ip, 40000), authkey=b'abracadabra')

    p_server = mp.Process(target=main_server, args=(qm_kwargs,))
    p_server.start()

    p_1 = mp.Process(target=main, args=(qm_kwargs,))
    p_1.start()

    p_2 = mp.Process(target=main_pmu, args=(qm_kwargs,))
    p_2.start()

    p_1.join()
    p_2.join()
    p_server.join()