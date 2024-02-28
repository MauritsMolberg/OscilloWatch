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
from topsrt.plotting.phasor_plots import VoltagePhasorPlot
from topsrt.interfacing import InterfacerQueuesThread
import numpy as np


class GenSpeedPlot(RTSimPlot):
    @staticmethod
    def get_init_data(rts):
        return rts.ps.gen['GEN'].n_units

    @staticmethod
    def read_input_signal(rts):
        return rts.sol.t, rts.ps.gen['GEN'].speed(rts.sol.x, rts.sol.v)



class VoltageAnglePlot(TimeWindowPlot, InterfacerQueuesThread):
    def __init__(self, rts=None, n_samples=100, update_freq=25, *args, **kwargs):
        self.n_samples = n_samples
        self.update_freq = update_freq
        self.subtract_mean = False
        InterfacerQueuesThread.__init__(self, rts)

    @staticmethod
    def get_init_data(rts):
         return rts.ps.n_bus, rts.ps.v_0

    def initialize(self, init_data):

        n_phasors, v_0 = init_data
        self.angles_prev = np.angle(v_0)
        self.angles_prev = np.unwrap(self.angles_prev)
        
        n_cols = n_phasors
        TimeWindowPlot.__init__(self, n_samples=self.n_samples, n_cols=n_cols, update_freq=self.update_freq)

    @staticmethod
    def read_input_signal(rts):
        return rts.sol.t, rts.ps.red_to_full.dot(rts.sol.v)

    def update(self, input):
        t, phasors = input
        angle = np.angle(phasors)
        angle = np.unwrap(np.vstack([self.angles_prev, angle]), axis=0)[1, :]
        self.angles_prev[:] = angle

        if self.subtract_mean:
            angle -= np.mean(angle)
        phasors_rot = abs(phasors)*np.exp(1j*angle)

        # Update internal states
        
        self.append(t, angle)


class VoltageMeanAnglePlot(TimeWindowPlot, InterfacerQueuesThread):
    def __init__(self, rts=None, n_samples=1000, update_freq=25, *args, **kwargs):
        self.n_samples = n_samples
        self.update_freq = update_freq
        self.subtract_mean = False
        InterfacerQueuesThread.__init__(self, rts)

    @staticmethod
    def get_init_data(rts):
         return rts.ps.n_bus, rts.ps.v_0

    def initialize(self, init_data):

        n_phasors, v_0 = init_data
        self.angles_prev = np.angle(v_0)
        self.angles_prev = np.unwrap(self.angles_prev)
        
        n_cols = n_phasors
        TimeWindowPlot.__init__(self, n_samples=self.n_samples, n_cols=1, update_freq=self.update_freq)

    @staticmethod
    def read_input_signal(rts):
        return rts.sol.t, rts.ps.red_to_full.dot(rts.sol.v)

    def update(self, input):
        t, phasors = input
        angle = np.angle(phasors)
        angle = np.unwrap(np.vstack([self.angles_prev, angle]), axis=0)[1, :]
        self.angles_prev[:] = angle
        
        mean_angle = np.mean(angle)
        if self.subtract_mean:
            angle -= mean_angle

        phasors_rot = abs(phasors)*np.exp(1j*angle)

        # Update internal states
        
        # self.append(t, mean_angle)
        self.append(t, mean_angle)
        # print(mean_angle)



def main():
    import tops.ps_models.k2a as model_data

    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    ps.ode_fun(0, ps.x0)
    rts = RealTimeSimulatorThread(ps, dt=5e-3, speed=1, solver=dps_sol.ModifiedEulerDAE)
    app = QtWidgets.QApplication(sys.argv)

    update_freq = 25
    tw_plot = GenSpeedPlot(rts=rts, n_samples=1000, update_freq=update_freq)
    sync_plot = SyncPlot(rts=rts, n_samples=1000, update_freq=update_freq)
    voltage_phasor_plot = VoltagePhasorPlot(rts, update_freq=update_freq)
    voltage_angle_plot = VoltageAnglePlot(rts, update_freq=update_freq)
    voltage_mean_angle_plot = VoltageMeanAnglePlot(rts, update_freq=update_freq)
    # sync_plot_2 = SyncPlot(n_samples=1000, update_freq=10)
    time.sleep(1)
    rts.start()
    tw_plot.start()
    sync_plot.start()
    voltage_phasor_plot.start()
    voltage_angle_plot.start()
    voltage_mean_angle_plot.start()


    # Add Control Widgets
    line_outage_ctrl = LineOutageWidget(rts)
    sin_ctrl = SimulationControl(rts)

    app.exec()

    return app


if __name__ == '__main__':
    main()