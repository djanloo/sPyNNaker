# Test PanHandler
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from run_manager import PanHandler, DataGatherer
from vogels_abbott import build_system

from local_utils import activity_stats, isi_stats, v_stats
from local_utils import  phase_invariant_average
from local_utils import synaptic_conductance_stats

import logging
from local_utils import set_loggers; set_loggers(lvl=logging.WARNING)
logging.getLogger("RUN_MANAGER").setLevel(logging.INFO)
logger=logging.getLogger("APPLICATION")
logger.setLevel(logging.DEBUG)

DURATION = 1000

default_system_params = dict(n_neurons=200, 
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
                    add_old = False
                    )

pan_handler = PanHandler(build_system)


for ts in [ .9,  1]:
    lunchbox_pars = default_lunchbox_params.copy()
    lunchbox_pars['timestep'] = ts
    lunchbox_pars['min_delay'] = int(lunchbox_pars['min_delay']/ts)*ts
    pan_handler.add_lunchbox_dict(lunchbox_pars)

for _ in range(2):
    for exc_conn_p in np.linspace(0.01, 0.4, 5):
        params = default_system_params.copy()
        params['exc_conn_p'] = exc_conn_p
        pan_handler.add_system_dict(params)

for ext in [activity_stats,
            isi_stats,
            synaptic_conductance_stats,
            v_stats, 
            # phase_invariant_average
            ]:

    pan_handler.add_extraction(ext)

pan_handler.run()

dg = DataGatherer(pan_handler.folder)
db = dg.gather()


