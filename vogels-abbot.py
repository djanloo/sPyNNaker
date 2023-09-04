"""Vogels-Abbot benchmark for test on integratos on spiNNaker

This is basically a python version of the jupyter notebook at 
https://github.com/albertoarturovergani/CNT-2023/blob/main/SpiNNaker/eg_balance-network.ipynb

AUTHOR: djanloo
DATE:   04/09/23
"""

import socket #?

from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility.plotting import Figure, Panel

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import logging
from local_utils import sim

dt = 1          # (ms) simulation timestep
tstop = 1000    # (ms) simulaton duration
delay = 2       # (ms) 

################ SETUP ################

sim.setup(
    timestep=dt, 
    min_delay=delay, 
    max_delay=delay) # [ms] # not that the max_delay supported by SpiNNaker is timestep * 144

rngseed = 98766987
rng = NumpyRNG(seed=rngseed, parallel_safe=True)


################ CELL STUFF ################

celltype = sim.IF_cond_exp

cell_params = dict(tau_m=20.0,# ms
                   tau_syn_E=5.0,# ms 
                   tau_syn_I=10.0,# ms
                   
                   v_rest=-60.0,# mV
                   v_reset=-60.0,# mV
                   v_thresh=-50.0,# mV
                   
                   cm=1.0,# µF/cm²
                   tau_refrac=5.0,# ms
                   i_offset=0.0,# nA

                    e_rev_E=0.0,# mV
                    e_rev_I=-80.0,# mV
                   )

################ POPULATIONS ################

n = 1500          # number of cells
r_ei = 4.0        # number of excitatory cells:number of inhibitory cells
n_exc = int(round((n * r_ei / (1 + r_ei))))  # number of excitatory cells
n_inh = n - n_exc                            # number of inhibitory cells
logging.info(f"Setting up a network with {n_exc} excitatory neurons and {n_inh} inhibitory neurons")

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
    logging.info("setting 50 neurons per core since we are on a spiNNaker machine")
    sim.set_number_of_neurons_per_core(sim.IF_cond_exp, 50)

################ SYNAPTIC STUFF & CONNECTIONS ################
w_exc = 4.0  *1e-3       # (uS)
w_inh = 51.0 *1e-3       # (uS) 

exc_synapses = sim.StaticSynapse(weight=w_exc, delay=delay)
inh_synapses = sim.StaticSynapse(weight=w_inh, delay=delay)

pconn = 0.02      # connection probability (2%)

exc_conn = sim.FixedProbabilityConnector(pconn, 
                                         #rng=rng # this raises ConfigurationException
                                         )
inh_conn = sim.FixedProbabilityConnector(pconn,
                                         #rng=rng # this raiss ConfigurationException
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
tstop = 100# (ms)
sim.run(tstop)

outputs = dict()

for layer in ['exc', 'inh']:
    
    # save on the notebook space
    outputs[layer] = pops[layer].get_data()
    
    # save in the folder space
    for recording in ['v', 'gsyn_inh', 'gsyn_exc', 'spikes']:
        import os
        try:
            os.mkdir("VA_results")
        except FileExistsError:
            pass
        pops[layer].write_data(f"VA_results/{layer}_{recording}.pkl")
        

