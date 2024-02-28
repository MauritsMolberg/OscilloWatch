import tops.dynamic as dps
import tops.solvers as dps_sol
from topsrt.sim import RealTimeSimulator, RealTimeSimulatorThread
import sys
from PySide6 import QtWidgets
from topsrt.gui import LineOutageWidget
from topsrt.rtsim_plot import RTSimPlot, SyncPlot
import pyqtgraph as pg
from pyqtgraph.console import ConsoleWidget
import numpy as np


def main():
    import tops.ps_models.ieee39 as model_data

    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    ps.ode_fun(0, ps.x0)
    rts = RealTimeSimulatorThread(ps, dt=5e-3, speed=1, solver=dps_sol.ModifiedEulerDAE)

    app = QtWidgets.QApplication(sys.argv)

    tw_plot = RTSimPlot(rts=rts, n_samples=1000)
    sync_plot = SyncPlot(rts=rts, n_samples=1000, update_freq=50)
    sync_plot_2 = SyncPlot(rts=rts, n_samples=1000, update_freq=10)
    # sync_plot_2 = SyncPlot(n_samples=1000, update_freq=10)

    rts.start()
    tw_plot.start()
    sync_plot.start()
    sync_plot_2.start()

    # Add Control Widgets
    line_outage_ctrl = LineOutageWidget(rts)

    c = ConsoleWidget(namespace={'pg': pg, 'np': np, 'rts': rts}, text='')
    c.show()

    app.exec()

    return app


if __name__ == '__main__':
    main()