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

import logging
from local_utils import set_loggers; set_loggers()
logger = logging.getLogger("RUNMANAGER")

from local_utils import get_sim
sim = get_sim()

################ PREAMBLE: CONSTANTS ################

DT = 1          # (ms)
DELAY = 2       # (ms)
TIMESCALE_FACTOR = 10

################ CELL PARAMETERS ################
CELLTYPE = sim.IF_cond_exp
CELL_PARAMS = dict(tau_m=20.0,# ms
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
################ SYNAPTIC STUFF & CONNECTIONS ################
W_EXC = 4.0  *1e-3       # (uS)
W_INH = 51.0 *1e-3       # (uS) 

################ THALAMIC STIMULUS ################
N_THALAMIC_CELLS = 20 
THALAMIC_STIM_DUR = 50.    # (ms) duration of random stimulation
THALAMIC_RATE = 100.       # (Hz) frequency of the random stimulation



################ RUN SCHEDULER ################
global pops
pops = dict()

def setup():
    """Sets up the simulation.
    
    NOTE: each subnetwork shares with the others the same simulation parameters.
    """
    sim.setup(
        timestep=DT,
        time_scale_factor=TIMESCALE_FACTOR,
        min_delay=DELAY, 
        # max_delay=delay # not supported
        )

def build_subnetwork(subnet_params):
    """Builds a subnetwork based on subnet params"""
    global pops
    subnet_idx = subnet_params['idx']
    logger.info(f"Setting subnet with id = {subnet_idx}")

    r_ei = 4.0
    n_exc = int(round((subnet_params['n_neurons'] * r_ei / (1 + r_ei))))
    n_inh = subnet_params['n_neurons'] - n_exc
    logger.info(f"Sub-network has {n_exc} excitatory neurons and {n_inh} inhibitory neurons")

    pops[f'exc_{subnet_idx}'] = sim.Population(
                                    n_exc, 
                                    CELLTYPE(**CELL_PARAMS), 
                                    label=f"exc_cells_{subnet_idx}")

    pops[f'inh_{subnet_idx}'] = sim.Population(
                                    n_inh, 
                                    CELLTYPE(**CELL_PARAMS), 
                                    label=f"inh_cells_{subnet_idx}")

    pops[f'exc_{subnet_idx}'].record(["spikes", 'v', 'gsyn_exc', 'gsyn_inh'])
    pops[f'inh_{subnet_idx}'].record(["spikes", 'v', 'gsyn_exc', 'gsyn_inh'])


    uniformDistr = RandomDistribution('uniform', 
                                    [CELL_PARAMS["v_reset"], CELL_PARAMS["v_thresh"]], 
                                    #   rng=rng # this causes a ConfigurationException
                                    )

    pops[f'exc_{subnet_idx}'].initialize(v=uniformDistr)
    pops[f'inh_{subnet_idx}'].initialize(v=uniformDistr)
    logger.info("Populations voltages initialized")

    if sim.__name__ == 'pyNN.spiNNaker':
        logger.info("Setting 50 neurons per core since we are on a spiNNaker machine")
        sim.set_number_of_neurons_per_core(CELLTYPE, 50)

    exc_synapses = sim.StaticSynapse(weight=W_EXC, delay=DELAY)
    inh_synapses = sim.StaticSynapse(weight=W_INH, delay=DELAY)

    exc_conn = sim.FixedProbabilityConnector(subnet_params['exc_conn_p'], 
                                            #  rng=rng # this raises ConfigurationException
                                            )
    logger.info(f"Initialized [green]excitatory[/green] FixedProbabilityConnector with p = {subnet_params['exc_conn_p']:.3}", extra=dict(markup=True))
    inh_conn = sim.FixedProbabilityConnector(subnet_params['inh_conn_p'],
                                            #  rng=rng # this raises ConfigurationException
                                            )
    logger.info(f"Initialized [blue]inhibitory[/blue] FixedProbabilityConnector with p = {subnet_params['exc_conn_p']:.3}", extra=dict(markup=True))

    connections = dict(
        
        e2e=sim.Projection(
                pops[f'exc_{subnet_idx}'],
                pops[f'exc_{subnet_idx}'], 
                exc_conn, 
                receptor_type='excitatory',
                synapse_type=exc_synapses),
            
        e2i=sim.Projection(
                pops[f'exc_{subnet_idx}'], 
                pops[f'inh_{subnet_idx}'], 
                exc_conn, 
                receptor_type='excitatory',
                synapse_type=exc_synapses),
        
        i2e=sim.Projection(
                pops[f'inh_{subnet_idx}'], 
                pops[f'exc_{subnet_idx}'], 
                inh_conn, 
                receptor_type='inhibitory',
                synapse_type=inh_synapses),
        
        i2i=sim.Projection(
                pops[f'inh_{subnet_idx}'],
                pops[f'inh_{subnet_idx}'],
                inh_conn, 
                receptor_type='inhibitory',
                synapse_type=inh_synapses)
        
    )
    logger.info("Connections Done")

    pops[f'thalamus_{subnet_idx}'] = sim.Population(
        N_THALAMIC_CELLS, 
        sim.SpikeSourcePoisson(rate=THALAMIC_RATE, duration=THALAMIC_STIM_DUR),
        label="expoisson")
    pops[f'thalamus_{subnet_idx}'].record("spikes")

    rconn = 0.01
    ext_conn = sim.FixedProbabilityConnector(rconn)

    connections[f'ext2e_{subnet_idx}'] = sim.Projection(
        pops[f'thalamus_{subnet_idx}'], 
        pops[f'exc_{subnet_idx}'], 
        ext_conn, 
        receptor_type='excitatory',
        synapse_type=sim.StaticSynapse(weight=0.1))

    connections[f'ext2i_{subnet_idx}'] = sim.Projection(
        pops[f'thalamus_{subnet_idx}'], 
        pops[f'inh_{subnet_idx}'], 
        ext_conn, 
        receptor_type='excitatory',
        synapse_type=sim.StaticSynapse(weight=0.1))
    
    logger.info(f"Thalamic stimulus Done")

def save(subnet_params_list):
    global pops
    try:
        os.mkdir("RM_results")
    except FileExistsError:
        pass

    for subnet_params in subnet_params_list:
        subnet_pops = [pn for pn in pops.keys() if pn.endswith(str(subnet_params['idx']))]
        logger.info(f"Found populations {subnet_pops} for idx={subnet_params['idx']}")

        # Saves results for the populations
        for subnet_pop in subnet_pops:
            logger.info(f"Saving for {subnet_pop}")
            pops[subnet_pop].write_data(f"RM_results/{subnet_pop}.pkl")

            # Saves the configuration of the run
            with open(f"RM_results/{subnet_pop}.cfg", 'wb') as file:
                logger.info(f"Saving config file for subnetwork {subnet_params['idx']} in RM_results/{subnet_pop}.cfg")
                pickle.dump(subnet_params, file)


if __name__ == "__main__":
    
    p_exc_conn = np.linspace(0.02, 0.08, 5)
    default_subnet_params = dict(n_neurons=500, 
                                 exc_conn_p=0.02, 
                                 inh_conn_p=0.02)
    logger.info(f"Default parameters are:\n{default_subnet_params}")
    logger.info(f"Scanning on exc_conn_p with values = {p_exc_conn}")
    params_list = []
    for pexc in p_exc_conn:
        subnet_params = default_subnet_params.copy()
        subnet_params['exc_conn_p'] = pexc
        subnet_params['idx'] =  hash(pexc)
        params_list.append(subnet_params)
        logger.info(f"Scheduled run with params {subnet_params}")


    setup()

    for subnet_params in params_list:
        build_subnetwork(subnet_params)
    
    logger.debug(f"Pops is {pops}")
    logger.debug(f"Pops keys are {pops.keys()}")

    logger.info("Starting simulation..")
    sim.run(1000)
    logger.info("Simulation Done")

    save(params_list)
