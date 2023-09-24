# Test PanHandler
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from run_manager import PanHandler, DataGatherer
from vogels_abbott import build_system

from local_utils import avg_activity

import logging
from local_utils import set_loggers; set_loggers(lvl=logging.WARNING)
logging.getLogger("RUN_MANAGER").setLevel(logging.DEBUG)
logger=logging.getLogger("APPLICATION")
logger.setLevel(logging.DEBUG)

DURATION = 1000

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
                    add_old = False
                    )

pan_handler = PanHandler(build_system)


for ts in [.1, .2, .3, .4, .5, .6, .7, .8, .9,  1]:
    lunchbox_pars = default_lunchbox_params.copy()
    lunchbox_pars['timestep'] = ts
    lunchbox_pars['min_delay'] = int(lunchbox_pars['min_delay']/ts)*ts
    pan_handler.add_lunchbox_dict(lunchbox_pars)

for _ in range(5):
    pan_handler.add_system_dict(default_system_params)

pan_handler.add_extraction(avg_activity)

pan_handler.run()


dg = DataGatherer(pan_handler.folder)
db = dg.gather()
logger.info(f"Database cols: {db.columns}")

plt.plot(db[(db["pop"] == 'exc')&(db.func == 'avg_activity')].timestep, db[(db['pop'] == 'exc')&(db.func == 'avg_activity')].extraction, ls="", marker=".")
plt.show()




