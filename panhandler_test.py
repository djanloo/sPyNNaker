# Test PanHandler
from run_manager import PanHandler
from vogels_abbott import build_system


from local_utils import avg_activity

import logging
from local_utils import set_loggers; set_loggers(lvl=logging.WARNING)
logging.getLogger("RUN_MANAGER").setLevel(logging.DEBUG)
logger=logging.getLogger("APPLICATION")
logger.setLevel(logging.DEBUG)

DURATION = 1000

default_system_params = dict(n_neurons=1000, 
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
                    folder=f"ush",
                    add_old = False
                    )

pan_handler = PanHandler(build_system)

for _ in range(3):
    for ts in [0.2, 0.5, 1]:
        lunchbox_pars = default_lunchbox_params.copy()
        lunchbox_pars['timestep'] = ts
        pan_handler.add_lunchbox_dict(lunchbox_pars)

pan_handler.add_system_dict(default_system_params)

pan_handler.add_extraction(avg_activity)

pan_handler.run()
logger.info(f"Extractions:\n{pan_handler.extractions}")


from run_manager import data_flattener
logger.info(data_flattener(pan_handler.extractions))

