from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import dynpssimpy.dynamic as dps
import dynpssimpy.solvers as dps_sol
import importlib
from emd import emd, plot_emd_results
from hht import calc_hilbert_spectrum, hht, plot_hilbert_spectrum
from hilbert_spectrum_stft import hilbert_spectrum_stft
importlib.reload(dps)



if __name__ == '__main__':

    # Load model
    import dynpssimpy.ps_models.k2a_no_controls as model_data
    model = model_data.load()

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    # ps.use_numba = True
    # Power flow calculation
    ps.power_flow()
    # Initialization
    ps.init_dyn_sim()
    #
    np.max(ps.ode_fun(0.0, ps.x0))
    # Specify simulation time
    #
    t_end = 20
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
    print('Ybus_full = ', ps.y_bus_red_full)
    print('Ybus_red = ', ps.y_bus_red)

    v_bus_mag = np.abs(ps.v_0)
    v_bus_angle = ps.v_0.imag / v_bus_mag
    #
    print(' ')
    print('Voltage magnitudes (p.u) = ', v_bus_mag)
    print(' ')
    print('Voltage angles     (rad) = ', v_bus_angle)
    print(' ')
    print('Voltage magnitudes  (kV) = ', v_bus_mag*[20, 20, 20, 20, 230, 230, 230, 230, 230, 230, 230])
    print(' ')
    # print(ps.v_n)
    print('v_vector = ', ps.v_0)
    print(' ')
    # print('Forskjell på red og full Ybus = ',ps.y_bus_red_full - ps.y_bus_red)
    #
    print('state description: ', ps.state_desc)
    print('Initial values on all state variables (G1 and IB) :')
    print(x0)
    print(' ')


    # Run simulation
    while t < t_end:
        # print(t)
        #v_bus_full = ps.red_to_full.dot(ps.v_red)
        # Simulate short circuit
        if 5 < t < 5.1:
            ps.y_bus_red_mod[7, 7] = 10000
        else:
            ps.y_bus_red_mod[7, 7] = 0
        # simulate a load change
        #if 2 < t < 4:
        #    ps.y_bus_red_mod[2, 2] = 0.1
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

    print("Q_e:", [round(num/900, 4) for num in Q_e_stored[0]] , [round(num/900, 4) for num in Q_e_stored[-1]] )
    print("V1:", round(V1_stored[0], 4), round(V1_stored[-1], 4))
    print("V2:", round(V2_stored[0], 4), round(V2_stored[-1], 4))
    print("V3:", round(V3_stored[0], 4), round(V3_stored[-1], 4))
    print("V4:", round(V4_stored[0], 4), round(V4_stored[-1], 4))

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    # Convert dict to pandas dataframe
    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)

    # Plot angle and speed
    fig, ax = plt.subplots(5)
    #fig.suptitle('Generator speed, angle and electric power')
    ax[0].plot(result[('Global', 't')], result.xs(key='speed', axis='columns', level=1))
    ax[0].set_ylabel('Speed (p.u.)')
    ax[1].plot(result[('Global', 't')], np.array(V1_stored))
    ax[1].plot(result[('Global', 't')], np.array(V2_stored))
    ax[1].plot(result[('Global', 't')], np.array(V3_stored))
    ax[1].plot(result[('Global', 't')], np.array(V4_stored))
    ax[1].set_ylabel("Terminal voltage (p.u.)")
    ax[2].plot(result[('Global', 't')], result.xs(key='angle', axis='columns', level=1))
    ax[2].set_ylabel('Angle (rad.)')
    ax[3].plot(result[('Global', 't')], np.array(P_e_stored)/[900, 900, 900, 900])
    ax[3].set_ylabel('Active power (p.u.)')
    ax[4].plot(result[('Global', 't')], np.array(Q_e_stored)/[900, 900, 900, 900])
    ax[4].set_ylabel('Reactive power (p.u.)')
    ax[4].set_xlabel('time (s)')


    plt.figure()
    plt.plot(result[('Global', 't')], np.array(V1_stored))
    plt.plot(result[('Global', 't')], np.array(V2_stored))
    plt.plot(result[('Global', 't')], np.array(V3_stored))
    plt.plot(result[('Global', 't')], np.array(V4_stored))
    plt.ylabel("Terminal voltage (p.u.)")
    plt.xlabel('time (s)')

    #plt.figure()
    #plt.plot(result[('Global', 't')], np.array(I_stored))
    #plt.xlabel('time (s)')
    #plt.ylabel('I_8-9 (magnitude p.u.)')

    #plt.figure()
    #plt.plot(result[('Global', 't')], np.array(E_f_stored))
    #plt.xlabel('time (s)')
    #plt.ylabel('E_q (p.u.)')



    imf_list, res = emd(V1_stored, max_imfs=4, remove_padding=True)
    plot_emd_results(V1_stored, imf_list, res, show=False)

    hilbert_spec, omegaAxis = hht(V1_stored, freq_resolution=1e-4, print_emd_time=True, print_hht_time=True, freq_tol="2fres")
    plot_hilbert_spectrum(hilbert_spec, omegaAxis, show=False)

    plt.show()
