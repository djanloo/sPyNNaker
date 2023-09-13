"""Vogels-Abbot benchmark for test on integratos on spiNNaker

This is basically a python version of the jupyter notebook at 
https://github.com/albertoarturovergani/CNT-2023/blob/main/SpiNNaker/eg_balance-network.ipynb

AUTHOR: djanloo
DATE:   04/09/23
"""
import numpy as np

import os
import matplotlib

if os.environ.get("USER") == "bbpnrsoa":
    matplotlib.use('agg')

import matplotlib.pyplot as plt
import matplotlib.tri as tri

from local_utils import get_sim
sim = get_sim()

from run_manager import RunBox, System
from vogels_abbott import build_system
from local_utils import avg_activity

import logging
from local_utils import set_loggers;

set_loggers(lvl=logging.WARNING) # Sets all loggers to warning
logging.getLogger("RUN_MANAGER").setLevel(logging.INFO) # Set Run Manager to info

logger = logging.getLogger("APPLICATION")
logger.setLevel(logging.DEBUG)

# Defines the RunBox where the systems will be runned on
runbox = RunBox(sim, timestep=1, 
                    time_scale_factor=50, 
                    duration=1000, 
                    min_delay=2
                )

# Default parameters of each system
default_params = dict(n_neurons=800, 
                      exc_conn_p=0.02, 
                      inh_conn_p=0.02,
                      synaptic_delay=2)


min_conn, max_conn = 0.01, 0.03

# Adds a bunch of systems t the runbox
for exc_conn_p in np.linspace(min_conn, max_conn, 3):
    for inh_conn_p in np.linspace(min_conn, max_conn, 3):
        params = default_params.copy()

        params['exc_conn_p'] = exc_conn_p
        params['inh_conn_p'] = inh_conn_p

        runbox.add_system(System(build_system, params))

# Here I specify which variables I want to compute for each population
# If the function cannot be evaluated a WARING will be raised
def mean_v(pop):
    return np.mean(pop.get_data('v').segments[0].analogsignals[0].magnitude)

def final_activity(pop):
    return avg_activity(pop, t_start=100)

# Here I tell the RunBox to extract mean_v for each population for each system
runbox.add_extraction(mean_v)
runbox.add_extraction(final_activity)

# Start the simulation & save
runbox.run()
runbox.save()
# runbox = RunBox.from_folder("RMv2")

# Extract the data in a format system_id -> function -> population -> values
results = runbox.get_extraction_triplets("exc_conn_p", "inh_conn_p", "final_activity")

logger.info(f"results is {results}")


# Create grid values first.
xi = np.linspace(min_conn, max_conn, 4)
yi = np.linspace(min_conn, max_conn, 4)

# Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
triang = tri.Triangulation(*(results['exc']['exc_conn_p', 'inh_conn_p'].T))
interpolator = tri.LinearTriInterpolator(triang, results['exc']['final_activity'])
Xi, Yi = np.meshgrid(xi, yi)
zi = interpolator(Xi, Yi)
plt.contourf(Xi,Yi,zi)
logger.info(zi)
plt.colorbar()

plt.savefig("runbox_test.png")
plt.show()