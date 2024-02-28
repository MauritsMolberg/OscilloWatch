import tops.dynamic as dps
import tops.solvers as dps_sol
from topsrt.sim import RealTimeSimulator
import threading
import time


if __name__ == '__main__':

    import tops.ps_models.ieee39 as model_data
    # import ps_models.ieee39 as model_data
    # import ps_models.sm_ib as model_data

    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    ps.ode_fun(0, ps.x0)
    rts = RealTimeSimulator(ps, dt=1e-3, speed=1, solver=dps_sol.ModifiedEulerDAE)
    # rts.sol.n_it = 0

    rts_thread = threading.Thread(target=rts.main_loop, daemon=True)
    rts_thread.start()

    while True:
        time.sleep(1)
        print(rts.dt_err)