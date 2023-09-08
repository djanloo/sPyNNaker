"""Vogels-Abbot benchmark for test on integratos on spiNNaker

This is basically a python version of the jupyter notebook at 
https://github.com/albertoarturovergani/CNT-2023/blob/main/SpiNNaker/eg_balance-network.ipynb

AUTHOR: djanloo
DATE:   04/09/23
"""
import pickle

from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility.plotting import Figure, Panel

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from local_utils import get_sim
sim = get_sim()

from local_utils import get_default_logger
logger = get_default_logger("simulation")

import argparse
parser = argparse.ArgumentParser(description='Vogels-Abbott benchmark')

parser.add_argument('--n_neurons', 
                    type=int,
                    default=1500, 
                    help='the number of total neurons neurons')

parser.add_argument('--exc_conn_p', 
                    type=float,
                    default=0.02, 
                    help='connection probability of excitatory network')

parser.add_argument('--inh_conn_p', 
                    type=float,
                    default=0.02, 
                    help='connection probability of inhibitory network')

parser.add_argument('--duration', 
                    type=int,
                    default=1000, 
                    help='the duration of the simularion in ms')

parser.add_argument('--out_prefix', 
                    type=str,
                    default="p", 
                    help='the prefix of the output files')

parser.add_argument('--timescale', 
                    type=int, 
                    default=1, 
                    help="the time scale factor")

run_params = parser.parse_args()

dt = 1          # (ms) simulation timestep
delay = 2       # (ms) 

################ SETUP ################

sim.setup(
    timestep=dt,
    time_scale_factor=run_params.timescale,
    min_delay=delay, 
    # max_delay=delay # not supported
    )

rngseed = 98766987
rng = None, NumpyRNG(seed=rngseed, parallel_safe=True)


################ CELL STUFF ################

celltype = sim.IF_cond_exp

cell_params = dict(tau_m=20.0,# ms
                   tau_syn_E=5.0,# ms 
                   tau_syn_I=10.0,# ms
                   
                   v_rest=-60.0,# mV
                   v_reset=-60.0,# mV
                   v_thresh=-50.0,# mV
                   
                   cm=0.2,#1.0,# µF/cm²
                   tau_refrac=5.0,# ms
                   i_offset=0.0,# nA

                    e_rev_E=0.0,# mV
                    e_rev_I=-80.0,# mV
                   )

################ POPULATIONS ################

r_ei = 4.0        # number of excitatory cells:number of inhibitory cells
n_exc = int(round((run_params.n_neurons * r_ei / (1 + r_ei))))  # number of excitatory cells
n_inh = run_params.n_neurons - n_exc                            # number of inhibitory cells
logger.info(f"Setting up a network with {n_exc} excitatory neurons and {n_inh} inhibitory neurons")

pops = {
    'exc': sim.Population(
                                n_exc, 
                                celltype(**cell_params), 
                                label="excitatory_cells"),

    'inh': sim.Population(
                                n_inh, 
                                celltype(**cell_params), 
                                label="inhibitory_cells")
}

pops['exc'].record(["spikes", 'v', 'gsyn_exc', 'gsyn_inh'])
pops['inh'].record(["spikes", 'v', 'gsyn_exc', 'gsyn_inh'])


uniformDistr = RandomDistribution('uniform', 
                                  [cell_params["v_reset"], cell_params["v_thresh"]], 
                                #   rng=rng # this causes a ConfigurationException
                                  )

pops['exc'].initialize(v=uniformDistr)
pops['inh'].initialize(v=uniformDistr)

if sim.__name__ == 'pyNN.spiNNaker':
    logger.info("setting 50 neurons per core since we are on a spiNNaker machine")
    sim.set_number_of_neurons_per_core(sim.IF_cond_exp, 50)

################ SYNAPTIC STUFF & CONNECTIONS ################
w_exc = 4.0  *1e-3       # (uS)
w_inh = 51.0 *1e-3       # (uS) 

exc_synapses = sim.StaticSynapse(weight=w_exc, delay=delay)
inh_synapses = sim.StaticSynapse(weight=w_inh, delay=delay)

exc_conn = sim.FixedProbabilityConnector(run_params.exc_conn_p, 
                                         rng=rng # this raises ConfigurationException
                                         )
inh_conn = sim.FixedProbabilityConnector(run_params.inh_conn_p,
                                         rng=rng # this raiss ConfigurationException
                                         )

connections = dict(
    
    e2e=sim.Projection(
            pops['exc'],
            pops['exc'], 
            exc_conn, 
            receptor_type='excitatory',
            synapse_type=exc_synapses),
        
    e2i=sim.Projection(
            pops['exc'], 
            pops['inh'], 
            exc_conn, 
            receptor_type='excitatory',
            synapse_type=exc_synapses),
    
    i2e=sim.Projection(
            pops['inh'], 
            pops['exc'], 
            inh_conn, 
            receptor_type='inhibitory',
            synapse_type=inh_synapses),
    
    i2i=sim.Projection(
            pops['inh'],
            pops['inh'],
            inh_conn, 
            receptor_type='inhibitory',
            synapse_type=inh_synapses)
    
)

################ THALAMIC STIMULUS ################
n_thalamic_cells = 20 
stim_dur = 50.    # (ms) duration of random stimulation
rate = 100.       # (Hz) frequency of the random stimulation

exc_conn = None

pops['thalamus'] = sim.Population(
    n_thalamic_cells, 
    sim.SpikeSourcePoisson(rate=rate, duration=stim_dur),
    label="expoisson")
pops['thalamus'].record("spikes")

rconn = 0.01
ext_conn = sim.FixedProbabilityConnector(rconn)

connections['ext2e'] = sim.Projection(
    pops['thalamus'], 
    pops['exc'], 
    ext_conn, 
    receptor_type='excitatory',
    synapse_type=sim.StaticSynapse(weight=0.1))

connections['ext2i'] = sim.Projection(
    pops['thalamus'], 
    pops['inh'], 
    ext_conn, 
    receptor_type='excitatory',
    synapse_type=sim.StaticSynapse(weight=0.1))

################ RUN ################
sim.run(run_params.duration)

outputs = dict()

import os

try:
    os.mkdir("VA_results")
except FileExistsError:
    pass

# Saves results for the populations
for pops_name in ['exc', 'inh']:
    pops[pops_name].write_data(f"VA_results/{run_params.out_prefix}_{pops_name}.pkl")

# Saves the configuration of the run
with open(f"VA_results/{run_params.out_prefix}.cfg", 'wb') as file:
    run_conf = vars(run_params)
    del run_conf["out_prefix"]
    pickle.dump(run_conf, file)
