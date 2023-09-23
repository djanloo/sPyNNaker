# Test PanHandler
from run_manager import PanHandler
from vogels_abbott import build_system

from local_utils import set_loggers; set_loggers()

DURATION = 500

default_system_params = dict(n_neurons=100, 
            exc_conn_p=0.03, 
            inh_conn_p=0.02,
            synaptic_delay=2
            )

default_lunchbox_params = dict(
                    timestep=1, 
                    time_scale_factor=50, # Defines the lunchbox where the systems will be runned on
                    duration=DURATION, 
                    min_delay=2,
                    # rng_seeds=[SEED],
                    neurons_per_core=250,
                    folder=f"panhandler_test",
                    add_old = False
                    )

pan_handler = PanHandler(build_system)

for ts in [0.2, 0.5, 1]:
    lunchbox_pars = default_lunchbox_params.copy()
    lunchbox_pars['timestep'] = ts
    pan_handler.add_lunchbox_dict(lunchbox_pars)

pan_handler.add_system_dict(default_system_params)
pan_handler.run()

