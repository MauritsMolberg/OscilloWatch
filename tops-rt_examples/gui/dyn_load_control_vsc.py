import tops.dynamic as dps
import tops.solvers as dps_sol
from topsrt.sim import RealTimeSimulator
import threading
import time
import sys
from PySide6 import QtWidgets
from topsrt.gui import LineOutageWidget, VSCControlWidget
from topsrt.time_window_plot import TimeWindowPlot
from topsrt.rtsim_plot import SyncPlot
from topsrt.plotting.phasor_plots import VoltagePhasorPlot
from topsrt.plotting.grid_plot import LiveGridPlot3D
from topsrt.rtsim_plot import RTSimPlot, SyncPlot


class LoadActivePowerPlot(RTSimPlot):
    @staticmethod
    def get_init_data(rts):
        return rts.ps.vsc['VSC'].n_units

    @staticmethod
    def read_input_signal(rts):
        return rts.sol.t, rts.ps.vsc['VSC'].P(rts.sol.x, rts.sol.v)
    
class LoadReactivePowerPlot(LoadActivePowerPlot):
    @staticmethod
    def read_input_signal(rts):
        return rts.sol.t, rts.ps.vsc['VSC'].Q(rts.sol.x, rts.sol.v)


def main(rts):
    update_freq = 25
    app = QtWidgets.QApplication(sys.argv)

    sync_plot = SyncPlot(rts=rts, n_samples=1000, update_freq=25)
    voltage_phasor_plot = VoltagePhasorPlot(rts, update_freq=25)
    grid_plot = LiveGridPlot3D(rts=rts, z_ax='angle')
    load_P_plot = LoadActivePowerPlot(rts=rts, n_samples=10000)
    load_Q_plot = LoadReactivePowerPlot(rts=rts, n_samples=10000)

    sync_plot.start()
    voltage_phasor_plot.start()
    grid_plot.start()
    load_P_plot.start()
    load_Q_plot.start()

    # Add Control Widgets
    line_outage_ctrl = LineOutageWidget(rts)
    load_control = VSCControlWidget(rts)

    app.exec()

    return app


if __name__ == '__main__':

    import tops.ps_models.k2a as model_data
    # import tops.ps_models.n44 as model_data

    model = model_data.load()
    # model['vsc'] = {'VSC': [
    #     ['name',    'T_pll',    'T_i',  'bus',  'P_K_p',    'P_K_i',    'Q_K_p',    'Q_K_i',    'P_setp',   'Q_setp'],
    #     *[[row[0],     0.1,        1,    row[1],     0.1,           0.1,       0.1,       0.1,        -row[2],      -row[3]] for row in model['loads'][1:]]
    # ]}
    # model['loads'] = {}

    import numpy as np
    target_load = 'L1'
    target_load_idx = np.argwhere([row[0] == target_load for row in model['loads']])[0][0]
    row = model['loads'].pop(target_load_idx)
    model['vsc'] = {'VSC': [
        ['name',    'T_pll',    'T_i',  'bus',  'P_K_p',    'P_K_i',    'Q_K_p',    'Q_K_i',    'P_setp',   'Q_setp'],
        [row[0],     0.1,        1,    row[1],     0.1,           0.1,       0.1,       0.1,        -row[2],      -row[3]]
    ]}

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    ps.ode_fun(0, ps.x0)
    rts = RealTimeSimulator(ps, dt=5e-3, speed=1, solver=dps_sol.ModifiedEulerDAE)

    rts_thread = threading.Thread(target=rts.main_loop, daemon=True)
    rts_thread.start()

    app = main(rts)