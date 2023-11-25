from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import dynpssimpy.dynamic as dps
import dynpssimpy.solvers as dps_sol
import importlib
importlib.reload(dps)

from methods.EMD import EMD
from methods.HHT import HHT
from methods.AnalysisSettings import AnalysisSettings
from methods.SegmentAnalysis import SegmentAnalysis
from methods.SignalAnalysis import SignalAnalysis


if __name__ == '__main__':

    # Load model
    import dynpssimpy.ps_models.n44_no_controls as model_data
    model = model_data.load()

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    # ps.use_numba = True
    # Power flow calculation
    ps.power_flow()
    # Initialization
    ps.init_dyn_sim()
    #    np.max(ps.ode_fun(0.0, ps.x0))
    # Specify simulation time
    #
    t_end = 32
    x0 = ps.x0.copy()
    # Add small perturbation to initial angle of first generator
    # x0[ps.gen_mdls['GEN'].state_idx['angle'][0]] += 1
    #
    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x0, t_end, max_step=5e-3)

    # Define other variables to plot
    P_e_stored = []
    Q_e_stored = []
    E_f_stored = []
    Igen= 0,0
    I_stored = []
    v_bus = []
    V1_stored = []
    V2_stored = []
    V3_stored = []
    V4_stored = []
    # Initialize simulation
    t = 0
    result_dict = defaultdict(list)
    t_0 = time.time()
    # ps.build_y_bus_red(ps.buses['name'])
    ps.build_y_bus(['B8'])
    #print('Ybus_full = ', ps.y_bus_red_full)
    #print('Ybus_red = ', ps.y_bus_red)

    v_bus_mag = np.abs(ps.v_0)
    v_bus_angle = ps.v_0.imag / v_bus_mag
    #
    #print(' ')
    #print('Voltage magnitudes (p.u) = ', v_bus_mag)
    #print(' ')
    #print('Voltage angles     (rad) = ', v_bus_angle)
    #print(' ')
    #print('Voltage magnitudes  (kV) = ', v_bus_mag*[20, 20, 20, 20, 230, 230, 230, 230, 230, 230, 230])
    #print(' ')
    # print(ps.v_n)
    #print('v_vector = ', ps.v_0)
    #print(' ')
    # print('Forskjell på red og full Ybus = ',ps.y_bus_red_full - ps.y_bus_red)
    #
    #print('state description: ', ps.state_desc)
    #print('Initial values on all state variables (G1 and IB) :')
    #print(x0)
    #print(' ')


    # Run simulation
    while t < t_end:
        # print(t)
        #v_bus_full = ps.red_to_full.dot(ps.v_red)
        # Simulate short circuit
        if 15 < t < 15.1:
            ps.y_bus_red_mod[0, 0] = 10000
        else:
            ps.y_bus_red_mod[0, 0] = 0
        # simulate a load change
        #if 2 < t < 4:
        #    ps.y_bus_red_mod[2, 2] = 0.05
        #else:
        #    ps.y_bus_red_mod[2, 2] = 0
        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t
        # Store result
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]
        Igen = ps.y_bus_red_full[7,8]*(v[8] -v[7])
        # Legger til nye outputs
        P_e_stored.append(ps.gen['GEN'].P_e(x, v).copy())
        Q_e_stored.append(ps.gen['GEN'].Q_e(x, v).copy())
        E_f_stored.append(ps.gen['GEN'].E_f(x, v).copy())
        V1_stored.append(abs(v[0]))
        V2_stored.append(abs(v[1]))
        V3_stored.append(abs(v[2]))
        V4_stored.append(abs(v[3]))
        I_stored.append(np.abs(Igen))


    #print("Q_e:", [round(num/900, 4) for num in Q_e_stored[0]] , [round(num/900, 4) for num in Q_e_stored[-1]] )
    #print("V1:", round(V1_stored[0], 4), round(V1_stored[-1], 4))
    #print("V2:", round(V2_stored[0], 4), round(V2_stored[-1], 4))
    #print("V3:", round(V3_stored[0], 4), round(V3_stored[-1], 4))
    #print("V4:", round(V4_stored[0], 4), round(V4_stored[-1], 4))

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    # Convert dict to pandas dataframe
    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)

    # Plot angle and speed
    fig, ax = plt.subplots(4)
    #fig.suptitle('Generator speed, angle and electric power')
    #ax[0].plot(result[('Global', 't')], result.xs(key='speed', axis='columns', level=1))
    #ax[0].set_ylabel('Speed (p.u.)')
    ax[0].plot(result[('Global', 't')], np.array(V1_stored))
    ax[0].plot(result[('Global', 't')], np.array(V2_stored))
    ax[0].plot(result[('Global', 't')], np.array(V3_stored))
    ax[0].plot(result[('Global', 't')], np.array(V4_stored))
    ax[0].set_ylabel("Terminal voltage (p.u.)")
    ax[1].plot(result[('Global', 't')], result.xs(key='angle', axis='columns', level=1))
    ax[1].set_ylabel('Angle (rad.)')
    ax[2].plot(result[('Global', 't')], np.array(P_e_stored))
    ax[2].set_ylabel('Active power (p.u.)')
    ax[3].set_xlabel('time (s)')
    ax[3].plot(result[('Global', 't')], result.xs(key='speed', axis='columns', level=1))
    ax[3].set_ylabel('Speed (p.u.)')


    plt.figure()
    plt.plot(result[('Global', 't')], np.array(V1_stored))
    plt.plot(result[('Global', 't')], np.array(V2_stored))
    plt.plot(result[('Global', 't')], np.array(V3_stored))
    plt.plot(result[('Global', 't')], np.array(V4_stored))
    plt.ylabel("Terminal voltage (p.u.)")
    plt.xlabel('time (s)')

    G1_speed = result.xs(key='speed', axis='columns', level=1)["G3000-1"].tolist()
    G1_speed_new = [G1_speed[i*4] for i in range(len(G1_speed)//4)]
    V1_new = [V1_stored[i*4] for i in range(len(V1_stored)//4)]
    #V1_new = moving_average(V1_new, 3)
    #np.save("k2a_with_controls_1s_fault_V.npy", V1_new)

    plt.figure()
    plt.plot(G1_speed_new)
    plt.ylabel("Bus 1 speed (p.u.)")
    plt.xlabel('time (s)')

    settings = AnalysisSettings(segment_length_time=10,
                                extra_padding_time_start=1,
                                extra_padding_time_end=1,
                                print_segment_analysis_time=True,
                                start_amp_curve_at_peak=False)

    sig_an = SignalAnalysis(V1_new, settings)
    sig_an.analyze_whole_signal()
    sig_an.write_results_to_file()

    for segment in sig_an.segment_analysis_list:
        segment.hht.emd.plot_emd_results(show=False)
        segment.hht.plot_hilbert_spectrum(show=False)

    plt.show()
