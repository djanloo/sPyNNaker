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
from local_utils import avg_activity, avg_isi_cv

import logging
from local_utils import set_loggers;

set_loggers(lvl=logging.WARNING) # Sets all loggers to warning
logging.getLogger("RUN_MANAGER").setLevel(logging.INFO) # Set Run Manager to info
logging.getLogger("UTILS").setLevel(logging.DEBUG)

logger = logging.getLogger("APPLICATION")
logger.setLevel(logging.DEBUG)

min_conn, max_conn = 0.01, 0.075
N = 4

# Default parameters of each system
default_params = dict(n_neurons=300, 
                      exc_conn_p=0.02, 
                      inh_conn_p=0.02,
                      synaptic_delay=2)

# Defines the RunBox where the systems will be runned on
runbox = RunBox(sim, timestep=1, 
                    time_scale_factor=50, 
                    duration=1000, 
                    min_delay=2,
                    neurons_per_core=250,
                    folder=f"n_{default_params['n_neurons']}_conn_scan"
                )

# Adds a bunch of systems t the runbox
for exc_conn_p in np.linspace(min_conn, max_conn, N):
    for inh_conn_p in np.linspace(min_conn, max_conn, N):
        params = default_params.copy()

        params['exc_conn_p'] = exc_conn_p
        params['inh_conn_p'] = inh_conn_p

        runbox.add_system(System(build_system, params))

# Here I specify which variables I want to compute for each population
# If the function cannot be evaluated a WARING will be raised
def mean_v(block):
    return np.mean(block.segments[0].filter(name="v")[0].signal.magnitude)

def final_activity(block):
    return avg_activity(block, t_start=100)

def final_isi_cv(block):
    return avg_isi_cv(block, t_start=100)

# Here I tell the RunBox to extract mean_v for each population for each system
runbox.add_extraction(mean_v)
runbox.add_extraction(final_activity)
runbox.add_extraction(final_isi_cv)

# Start the simulation & save
runbox.run()
runbox.extract_and_save()

runbox = RunBox.from_folder(f"n_{default_params['n_neurons']}_conn_scan")

levels = [np.linspace(-0.1, 80, 20), np.linspace(-0.1, 2, 10) ]
for extraction, lvls in zip(["final_activity", "final_isi_cv"], levels):
    plt.figure()
    # Extract the data in a format system_id -> function -> population -> values
    results = runbox.get_extraction_triplets("exc_conn_p", "inh_conn_p", extraction)

    logger.info(f"results is {results}")


    # Create grid values first.
    min_conn, max_conn = np.min(results['exc']['exc_conn_p', 'inh_conn_p']), np.max(results['exc']['exc_conn_p', 'inh_conn_p'])
    xi = np.linspace(min_conn, max_conn, 3*N)
    yi = np.linspace(min_conn, max_conn, 3*N)

    # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
    triang = tri.Triangulation(*(results['exc']['exc_conn_p', 'inh_conn_p'].T))
    interpolator = tri.LinearTriInterpolator(triang, results['exc'][extraction])
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)
    mappable=plt.contourf(Xi,Yi,zi, levels=lvls)
    logger.info(zi)
    
    plt.scatter(*(results['exc']['exc_conn_p', 'inh_conn_p'].T), 
                c=results['exc'][extraction], edgecolor="k",vmin=lvls[0], vmax=lvls[-1])
    
    plt.colorbar(mappable)

    plt.title(f"{extraction} for n_neurons={default_params['n_neurons']}")
    
    plt.xlabel("Excitatory connectivity")
    plt.ylabel("Inhibitory connectivity")
    plt.savefig(f"{runbox.folder}/{extraction}.png")
plt.show()