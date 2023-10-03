from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import dynpssimpy.dynamic as dps
import dynpssimpy.solvers as dps_sol
import importlib
importlib.reload(dps)


if __name__ == '__main__':

    # Load model
    import dynpssimpy.ps_models.assignment_model as model_data
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
    t_end = 5
    x0 = ps.x0.copy()
    # Add small perturbation to initial angle of first generator
    # x0[ps.gen_mdls['GEN'].state_idx['angle'][0]] += 1
    #
    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x0, t_end, max_step=5e-3)

    # Define other variables to plot
    P_e_stored = []
    P_m_stored = []
    E_f_stored = []
    I_stored = []
    v_bus = []
    # Initialize simulation
    t = 0
    result_dict = defaultdict(list)
    t_0 = time.time()
    # ps.build_y_bus_red(ps.buses['name'])
    ps.build_y_bus(['B2'])

    v_bus_mag = np.abs(ps.v_0)
    v_bus_angle = ps.v_0.imag / v_bus_mag
    #
    print(' ')
    print('Voltage magnitudes (p.u) = ', v_bus_mag)
    print('Voltage angles     (rad) = ', v_bus_angle)
    print(' ')
    print('Voltage magnitudes  (kV) = ', v_bus_mag*[10, 245, 245])
    print(' ')
    # print(ps.v_g)
    #
    print('state description: ', ps.state_desc)
    print('Initial values on all state variables (G1 and IB) :')
    print(x0)
    print(' ')

    event_flag_disconnect = True
    event_flag_connect = True

    #Timings
    fault_start_time = 1
    fault_clearing_time = .2
    reconnection_time = .1

    # Run simulation
    while t < t_end:
        # Simulate short circuit
        if fault_start_time < t < fault_start_time + fault_clearing_time:
            ps.y_bus_red_mod[1, 1] = 10000
        else:
            ps.y_bus_red_mod[1, 1] = 0

        # Line disconnect
        if fault_start_time + fault_clearing_time < t < fault_start_time + fault_clearing_time + reconnection_time and event_flag_disconnect:
            event_flag_disconnect = False
            ps.lines['Line'].event(ps, ps.lines['Line'].par['name'][0], 'disconnect')

        # Line connect
        if t > fault_start_time + fault_clearing_time + reconnection_time and event_flag_connect:
            event_flag_connect = False
            ps.lines['Line'].event(ps, ps.lines['Line'].par['name'][0], 'connect')

        # simulate a load change
        # if 2 < t < 4:
        #    ps.y_bus_red_mod[2, 2] = 0.1
        # else:
        #   ps.y_bus_red_mod[2, 2] = 0

        # print(t)
        # v_bus_full = ps.red_to_full.dot(ps.v_red)

        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        # Store result
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]
        # Legger til nye outputs
        P_e_stored.append(ps.gen['GEN'].P_e(x, v).copy())
        P_m_stored.append(ps.gen['GEN'].P_m(x, v).copy())
        E_f_stored.append(ps.gen['GEN'].E_f(x, v).copy())

        I_gen = ps.y_bus_red_full[0, 1] * (v[1] - v[0])
        I_stored.append(np.abs(I_gen))

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    # Convert dict to pandas dataframe
    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)

    # Plot angle and speed
    fig, ax = plt.subplots(3)
    #fig.suptitle('Generator speed, angle and electric power')
    ax[0].plot(result[('Global', 't')], result.xs(key='speed', axis='columns', level=1))
    ax[0].set_ylabel('Speed (p.u.)')
    #ax[1].plot(result[('Global', 't')], result.xs(key='angle', axis='columns', level=1))
    ax[1].plot(result[('Global', 't')], np.array(P_m_stored))
    ax[1].set_ylabel('Mec. power (p.u.)')
    ax[2].plot(result[('Global', 't')], np.array(P_e_stored)/[50, 10000])
    ax[2].set_ylabel('El. power (p.u.)')
    #ax[3].plot(result[('Global', 't')], np.array(I_stored))
    ax[2].set_xlabel('time (s)')
    #ax[3].set_ylabel('I_g (pu)')

    print(ps.y_bus_red_mod)

    plt.figure()
    plt.plot(result[('Global', 't')], np.array(E_f_stored))
    plt.xlabel('time (s)')
    plt.ylabel('E_q (p.u.)')

    plt.show()
