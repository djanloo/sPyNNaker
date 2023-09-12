"""Vogels-Abbot benchmark for test on integratos on spiNNaker

This is basically a python version of the jupyter notebook at 
https://github.com/albertoarturovergani/CNT-2023/blob/main/SpiNNaker/eg_balance-network.ipynb

AUTHOR: djanloo
DATE:   04/09/23
"""
import os
import pickle
import numpy as np

from pyNN.random import RandomDistribution

from local_utils import get_sim, num
sim = get_sim()

import logging
from local_utils import set_loggers; set_loggers()


from run_manager import RunBox, System
logger = logging.getLogger("mp_va")

from vogels_abbott import build_system


# Here I specify which variables I want to compute
def mean_v(pop):
    return np.mean(pop.get_v().segments[0].analogsignals[0].magnitude, axis=1)


default_params = dict(n_neurons=100, 
                      exc_conn_p=0.02, 
                      inh_conn_p=0.02,
                      synaptic_delay=2)

a = System(build_system, default_params)

modified_params = default_params.copy()
modified_params['n_neurons'] = 150

b = System(build_system, modified_params)

runbox = RunBox(sim, dict(timestep=1, 
                    time_scale_factor=10, 
                    duration=100, 
                    min_delay=2)
                )
runbox.add_system(a)
runbox.add_system(b)

runbox.add_extraction(mean_v)

runbox.run()
runbox.save()

results = runbox.extract()

logger.info(f"results is {results}")

# from analysis import runbox_analysis

# runbox_analysis({'folder': 'RMv2', 
#                  'plot': ['spikes', 'density', 'quantiles'], 
#                  'bins': 30, 
#                  'quantity': 'v', 
#                  'v': 1, 
#                  'list_files': False, 
#                  'all':True, 
#                  'conf': None })