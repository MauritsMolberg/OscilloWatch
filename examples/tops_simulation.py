"""
Note: This script requires TOPS in order to run:
https://github.com/hallvar-h/TOPS

TOPS is not listed in requirements.txt and must be installed manually with pip, using the command "pip install tops" in
a terminal.
"""

import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time

import numpy as np
import tops.dynamic as dps
import tops.solvers as dps_sol

from OscilloWatch.OWSettings import OWSettings
from OscilloWatch.SignalSnapshotAnalysis import SignalSnapshotAnalysis

if __name__ == '__main__':

    # Load model
    import tops.ps_models.k2a as model_data
    model = model_data.load()

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    t_end = 32
    x_0 = ps.x_0.copy()

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

    # Initialize simulation
    t = 0
    res = defaultdict(list)
    t_0 = time.time()

    # Initialize lists for variables to store
    V1 = []
    V2 = []
    V3 = []
    V4 = []

    theta1 = []
    theta2 = []
    theta3 = []
    theta4 = []

    # Event settings
    short_circuit_enable = False
    line_outage_event_flag = True
    line_reconnect_event_flag = True

    event_start_time = 17
    fault_clearing_time = .2  # Time taken to clear short circuit or reconnect line

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        # Short circuit
        if event_start_time <= t <= event_start_time + fault_clearing_time and short_circuit_enable:
            ps.y_bus_red_mod[0, 0] = 1e6
        else:
            ps.y_bus_red_mod[0, 0] = 0


        # Line disconnect
        if t > event_start_time and line_outage_event_flag:
            line_outage_event_flag = False
            ps.lines['Line'].event(ps, ps.lines['Line'].par['name'][0], 'disconnect')

        # Line reconnect
        if t > event_start_time + fault_clearing_time and line_reconnect_event_flag:
            line_reconnect_event_flag = False
            ps.lines['Line'].event(ps, ps.lines['Line'].par['name'][0], 'connect')

        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        dx = ps.ode_fun(0, ps.x_0)

        # Store result
        res['t'].append(t)
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
        V1.append(abs(v[0]))
        V2.append(abs(v[1]))
        V3.append(abs(v[2]))
        V4.append(abs(v[3]))
        theta1.append(np.angle(v[0]))
        theta2.append(np.angle(v[1]))
        theta3.append(np.angle(v[2]))
        theta4.append(np.angle(v[3]))

    print('\nSimulation completed in {:.2f} seconds.'.format(time.time() - t_0))


    # Downsample speed measurements to 50 Hz and store in separate lists
    G1_speed = [round(res["gen_speed"][i][0], 10) for i in range(0, len(res["gen_speed"]), 4)]
    G2_speed = [round(res["gen_speed"][i][1], 10) for i in range(0, len(res["gen_speed"]), 4)]
    G3_speed = [round(res["gen_speed"][i][2], 10) for i in range(0, len(res["gen_speed"]), 4)]
    G4_speed = [round(res["gen_speed"][i][3], 10) for i in range(0, len(res["gen_speed"]), 4)]

    # Downsample voltage measurements to 50 Hz
    V1 = [round(V1[i], 10) for i in range(0, len(V1), 4)]
    V2 = [round(V2[i], 10) for i in range(0, len(V2), 4)]
    V3 = [round(V3[i], 10) for i in range(0, len(V3), 4)]
    V4 = [round(V4[i], 10) for i in range(0, len(V4), 4)]

    theta1 = [round(theta1[i], 10) for i in range(0, len(theta1), 4)]
    theta2 = [round(theta2[i], 10) for i in range(0, len(theta2), 4)]
    theta3 = [round(theta3[i], 10) for i in range(0, len(theta3), 4)]
    theta4 = [round(theta4[i], 10) for i in range(0, len(theta4), 4)]


    settings = OWSettings(
                          fs=50,
                          segment_length_time=10,
                          extension_padding_time_start=10,
                          extension_padding_time_end=2,
                          max_imfs=5,
                          skip_storing_uncertain_modes=False,
                          minimum_frequency=0.1,
                          include_asterisk_explanations=False,
    )

    settings.results_file_path = "../results/TOPS/G1_speed"
    snap_an = SignalSnapshotAnalysis(G1_speed, settings)
    snap_an.analyze_whole_signal()
    # snap_an.segment_analysis_list[0].hht.emd.plot_emd_results()
    # snap_an.segment_analysis_list[0].hht.plot_hilbert_spectrum()
    # snap_an.segment_analysis_list[1].hht.emd.plot_emd_results()
    # snap_an.segment_analysis_list[1].hht.plot_hilbert_spectrum()

    settings.results_file_path = "../results/TOPS/G2_speed"
    snap_an = SignalSnapshotAnalysis(G2_speed, settings)
    snap_an.analyze_whole_signal()

    settings.results_file_path = "../results/TOPS/G3_speed"
    snap_an = SignalSnapshotAnalysis(G3_speed, settings)
    snap_an.analyze_whole_signal()

    settings.results_file_path = "../results/TOPS/G4_speed"
    snap_an = SignalSnapshotAnalysis(G4_speed, settings)
    snap_an.analyze_whole_signal()


    settings.results_file_path = "../results/TOPS/V1"
    snap_an = SignalSnapshotAnalysis(V1, settings)
    snap_an.analyze_whole_signal()

    settings.results_file_path = "../results/TOPS/V2"
    snap_an = SignalSnapshotAnalysis(V2, settings)
    snap_an.analyze_whole_signal()

    settings.results_file_path = "../results/TOPS/V3"
    snap_an = SignalSnapshotAnalysis(V3, settings)
    snap_an.analyze_whole_signal()

    settings.results_file_path = "../results/TOPS/V4"
    snap_an = SignalSnapshotAnalysis(V4, settings)
    snap_an.analyze_whole_signal()

    settings.results_file_path = "../results/TOPS/theta1"
    snap_an = SignalSnapshotAnalysis(theta1, settings)
    snap_an.analyze_whole_signal()
    snap_an.segment_analysis_list[0].hht.emd.plot_emd_results()
    snap_an.segment_analysis_list[0].hht.plot_hilbert_spectrum()
    snap_an.segment_analysis_list[1].hht.emd.plot_emd_results()
    snap_an.segment_analysis_list[1].hht.plot_hilbert_spectrum()

    settings.results_file_path = "../results/TOPS/theta2"
    snap_an = SignalSnapshotAnalysis(theta2, settings)
    snap_an.analyze_whole_signal()

    settings.results_file_path = "../results/TOPS/theta3"
    snap_an = SignalSnapshotAnalysis(theta3, settings)
    snap_an.analyze_whole_signal()

    settings.results_file_path = "../results/TOPS/theta4"
    snap_an = SignalSnapshotAnalysis(theta4, settings)
    snap_an.analyze_whole_signal()




    t_axis_50Hz = np.linspace(0, t_end, len(G1_speed))

    plt.figure()
    plt.plot(t_axis_50Hz, G1_speed, label="Gen. 1")
    plt.plot(t_axis_50Hz, G2_speed, label="Gen. 2")
    plt.plot(t_axis_50Hz, G3_speed, label="Gen. 3")
    plt.plot(t_axis_50Hz, G4_speed, label="Gen. 4")
    plt.xlabel('Time [s]')
    plt.ylabel('Gen. speed')
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(t_axis_50Hz, V1, label="Bus 1")
    plt.plot(t_axis_50Hz, V2, label="Bus 2")
    plt.plot(t_axis_50Hz, V3, label="Bus 3")
    plt.plot(t_axis_50Hz, V4, label="Bus 4")
    plt.xlabel('Time [s]')
    plt.ylabel('Bus voltage magnitude [pu]')
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(t_axis_50Hz, theta1, label="Bus 1")
    plt.plot(t_axis_50Hz, theta2, label="Bus 2")
    plt.plot(t_axis_50Hz, theta3, label="Bus 3")
    plt.plot(t_axis_50Hz, theta4, label="Bus 4")
    plt.xlabel('Time [s]')
    plt.ylabel('Bus voltage angle [rad]')
    plt.legend()
    plt.tight_layout()


    plt.show()
