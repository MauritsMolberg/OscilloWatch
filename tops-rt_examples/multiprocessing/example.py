import multiprocessing as mp
from topsrt.interfacing import QueueManager, InterfaceListener
from PySide6 import QtWidgets
from topsrt.rtsim_plot import SyncPlot
import sys
from topsrt.rtsim_plot import RTSimPlot
import tops.dynamic as dps
from topsrt.sim import RealTimeSimulatorThread
from topsrt.gui import LineOutageWidget, ConsoleWidget
from topsrt.plotting.phasor_plots import VoltagePhasorPlot, GenPhasorPlot
from topsrt.plotting.grid_plot import LiveGridPlot3D


def main_pod(qm_kwargs):
    manager = QueueManager(**qm_kwargs)
    manager.connect()

    app = QtWidgets.QApplication(sys.argv)
    # interface = RTSimPlot(n_samples=1000)
    # InterfaceListener.send_interface_init(manager, interface)

    sync_plot = SyncPlot(n_samples=1000, update_freq=50)
    sync_plot_2 = SyncPlot(n_samples=1000, update_freq=10)
    tw_plot = RTSimPlot(n_samples=1000)
    voltage_phasor_plot = VoltagePhasorPlot(update_freq=50)
    gen_phasor_plot = GenPhasorPlot(update_freq=50)
    grid_plot = LiveGridPlot3D(z_ax='angle', use_colors=True)

    [InterfaceListener.send_interface_init(manager, interface) for interface in [sync_plot, sync_plot_2, tw_plot, voltage_phasor_plot, gen_phasor_plot, grid_plot]]
    sync_plot.start()
    sync_plot_2.start()
    tw_plot.start()
    voltage_phasor_plot.start()
    gen_phasor_plot.start()
    grid_plot.start()

    app.exec()

    return app


def main(qm_kwargs):

    manager = QueueManager(**qm_kwargs)
    manager.connect()
    init_queue = manager.get_init_queue()
    interface_listener = InterfaceListener(init_queue)

    import tops.ps_models.ieee39 as model_data

    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    ps.ode_fun(0, ps.x0)
    rts = RealTimeSimulatorThread(ps, dt=10e-3, speed=1)

    interface_listener.connect(rts)
    interface_listener.start()

    # GUI
    update_freq = 25
    app = QtWidgets.QApplication(sys.argv)

    # Add Control Widgets
    line_outage_ctrl = LineOutageWidget(rts)
    c = ConsoleWidget(namespace={'rts': rts}, text='')
    c.show()

    rts.start()
    app.exec()
    rts.stop()

    return app


def main_server(qm_kwargs):
    manager = QueueManager(server=True, **qm_kwargs)
    manager.start()


if __name__ == '__main__':
    # qm_kwargs = dict(address=('10.0.0.16', 50000), authkey=b'abracadabra')
    # qm_kwargs = dict(address=('192.168.11.110', 50000), authkey=b'abracadabra')
    import socket
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    port = 50000
    qm_kwargs = dict(address=(ip, port), authkey=b'abracadabra')

    p_server = mp.Process(target=main_server, args=(qm_kwargs,))
    p_server.start()

    p_1 = mp.Process(target=main, args=(qm_kwargs,))
    p_1.start()

    p_2 = mp.Process(target=main_pod, args=(qm_kwargs,))
    p_2.start()

    p_1.join()
    p_2.join()
    p_server.join()