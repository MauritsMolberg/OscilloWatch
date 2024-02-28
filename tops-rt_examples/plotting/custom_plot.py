import tops.dynamic as dps
import tops.solvers as dps_sol
from topsrt.sim import RealTimeSimulator, RealTimeSimulatorThread
import threading
import time
import sys
from PySide6 import QtWidgets
from topsrt.gui import LineOutageWidget, SimulationControl
from topsrt.time_window_plot import TimeWindowPlot
from topsrt.rtsim_plot import RTSimPlot, SyncPlot


class GenSpeedPlot(RTSimPlot):
    @staticmethod
    def get_init_data(rts):
        return rts.ps.gen['GEN'].n_units

    @staticmethod
    def read_input_signal(rts):
        return rts.sol.t, rts.ps.gen['GEN'].speed(rts.sol.x, rts.sol.v)


def main():
    import tops.ps_models.ieee39 as model_data

    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    ps.ode_fun(0, ps.x0)
    rts = RealTimeSimulatorThread(ps, dt=5e-3, speed=1, solver=dps_sol.ModifiedEulerDAE)
    app = QtWidgets.QApplication(sys.argv)

    tw_plot = GenSpeedPlot(rts=rts, n_samples=1000, update_freq=10)
    sync_plot = SyncPlot(rts=rts, n_samples=1000, update_freq=25)
    # sync_plot_2 = SyncPlot(n_samples=1000, update_freq=10)

    rts.start()
    tw_plot.start()
    sync_plot.start()

    # Add Control Widgets
    line_outage_ctrl = LineOutageWidget(rts)
    sin_ctrl = SimulationControl(rts)

    app.exec()

    return app


if __name__ == '__main__':
    main()