"""Vogels-Abbot benchmark for test on integratos on spiNNaker

This is basically a python version of the jupyter notebook at 
https://github.com/albertoarturovergani/CNT-2023/blob/main/SpiNNaker/eg_balance-network.ipynb

AUTHOR: djanloo
DATE:   04/09/23
"""
import os
import numpy as np

from local_utils import get_sim
sim = get_sim()

import logging
from local_utils import set_loggers; set_loggers()

from run_manager import RunBox, System
logger = logging.getLogger("mp_va")

from vogels_abbott import build_system
from local_utils import avg_activity

import matplotlib.pyplot as plt

# Defines the RunBox where the systems will be runned on
runbox = RunBox(sim, dict(timestep=1, 
                    time_scale_factor=10, 
                    duration=1000, 
                    min_delay=2)
                )

# Default parameters of each system
default_params = dict(n_neurons=100, 
                      exc_conn_p=0.02, 
                      inh_conn_p=0.02,
                      synaptic_delay=2)
for _ in range(2):
    for n in [100, 300, 600, 750, 900, 1000]:
        params = default_params.copy()
        params['n_neurons'] = n

        runbox.add_system(System(build_system, params))

# Here I specify which variables I want to compute for each population
# If the function cannot be avaluated a WARING will be raised
def mean_v(pop):
    return np.mean(pop.get_v().segments[0].analogsignals[0].magnitude)

def final_activity(pop):
    return avg_activity(pop, t_start=500)

# Here I tell the RunBox to extract mean_v for each population for each system
runbox.add_extraction(mean_v)
runbox.add_extraction(final_activity)

# Start the simulation & save
runbox.run()
runbox.save()

# Extract the data in a format system_id -> function -> population -> values
results = runbox.get_extraction_couple("n_neurons", "exc", "final_activity")

logger.info(f"results is {results}")
plt.plot(results['n_neurons'], results['final_activity'], marker=".", ls="")

plt.show()
# from analysis import runbox_analysis

# runbox_analysis({'folder': 'RMv2', 
#                  'plot': ['spikes', 'density', 'quantiles'], 
#                  'bins': 30, 
#                  'quantity': 'v', 
#                  'v': 1, 
#                  'list_files': False, 
#                  'all':True, 
#                  'conf': None })