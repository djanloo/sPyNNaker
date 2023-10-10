"""Vogels-Abbot benchmark for test on integratos on spiNNaker

This is basically a python version of the jupyter notebook at 
https://github.com/albertoarturovergani/CNT-2023/blob/main/SpiNNaker/eg_balance-network.ipynb

AUTHOR: djanloo
DATE:   04/09/23
"""
from pyNN.random import NumpyRNG, RandomDistribution

from local_utils import get_sim
sim = get_sim()

import logging
from local_utils import set_loggers; set_loggers()
logger = logging.getLogger("NETWORK_BUILDING")

RNGSEED = 98766987
rng = NumpyRNG(seed=RNGSEED, parallel_safe=True)

############ CONSTANTS #############
CELLTYPE = sim.IF_cond_exp
CELL_PARAMS = dict(tau_m=20.0,# ms
                    tau_syn_E=5.0,# ms 
                    tau_syn_I=10.0,# ms
                    
                    v_rest=-60.0,# mV
                    v_reset=-61.0,# mV
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
THALAMIC_CONNECTIVITY = 0.1

def build_system(system_params):
    """Builds the system with parameters specified in system_params.
    
    Returns a dict of populations.
    """
    pops = dict()

    r_ei = 4.0
    n_exc = int(round((system_params['n_neurons'] * r_ei / (1 + r_ei))))
    n_inh = system_params['n_neurons'] - n_exc
    logger.info(f"Sub-network has {n_exc} (type {type(n_exc)}) excitatory neurons and {n_inh} (type {type(n_inh)}) inhibitory neurons")

    pops[f'exc'] = sim.Population(
                                    n_exc, 
                                    CELLTYPE(**CELL_PARAMS), 
                                    label=f"exc_cells")
    pops[f'inh'] = sim.Population(
                                    n_inh, 
                                    CELLTYPE(**CELL_PARAMS), 
                                    label=f"inh_cells")
    # try:
    #     pops['exc'].set(seed=RNGSEED)
    #     pops['inh'].set(seed=RNGSEED)
    # except Exception as e:
    #     logger.warn(f"Exception raised when setting the seed of populations: {e}")

    pops[f'exc'].record(["spikes", 'v', 'gsyn_exc', 'gsyn_inh'])
    pops[f'inh'].record(["spikes", 'v', 'gsyn_exc', 'gsyn_inh'])
    # pops[f'exc'].record(["spikes", 'v'])
    # pops[f'inh'].record(["spikes", 'v'])

    uniformDistr = RandomDistribution('uniform', 
                                    [CELL_PARAMS["v_reset"], CELL_PARAMS["v_thresh"]], 
                                    #   rng=rng # this causes a ConfigurationException
                                    )

    pops[f'exc'].initialize(v=uniformDistr)
    pops[f'inh'].initialize(v=uniformDistr)
    logger.info("Populations voltages initialized")

    exc_synapses = sim.StaticSynapse(weight=W_EXC, delay=system_params['synaptic_delay'])
    inh_synapses = sim.StaticSynapse(weight=W_INH, delay=system_params['synaptic_delay'])

    exc_conn = sim.FixedProbabilityConnector(system_params['exc_conn_p'], 
                                            #  rng=rng # this raises ConfigurationException
                                            )
    logger.info(f"Initialized [green]excitatory[/green] FixedProbabilityConnector with p = {system_params['exc_conn_p']:.3}", extra=dict(markup=True))
    inh_conn = sim.FixedProbabilityConnector(system_params['inh_conn_p'],
                                            #  rng=rng # this raises ConfigurationException
                                            )
    logger.info(f"Initialized [blue]inhibitory[/blue] FixedProbabilityConnector with p = {system_params['exc_conn_p']:.3}", extra=dict(markup=True))

    connections = dict(
        
        e2e=sim.Projection(
                pops[f'exc'],
                pops[f'exc'], 
                exc_conn, 
                receptor_type='excitatory',
                synapse_type=exc_synapses),
            
        e2i=sim.Projection(
                pops[f'exc'], 
                pops[f'inh'], 
                exc_conn, 
                receptor_type='excitatory',
                synapse_type=exc_synapses),
        
        i2e=sim.Projection(
                pops[f'inh'], 
                pops[f'exc'], 
                inh_conn, 
                receptor_type='inhibitory',
                synapse_type=inh_synapses),
        
        i2i=sim.Projection(
                pops[f'inh'],
                pops[f'inh'],
                inh_conn, 
                receptor_type='inhibitory',
                synapse_type=inh_synapses)
        
    )
    logger.info("Connections Done")

    pops[f'thalamus'] = sim.Population(
        N_THALAMIC_CELLS, 
        sim.SpikeSourcePoisson(rate=THALAMIC_RATE, duration=THALAMIC_STIM_DUR),
        label="expoisson")
    # pops[f'thalamus'].record("spikes")

    rconn = 0.01
    ext_conn = sim.FixedProbabilityConnector(THALAMIC_CONNECTIVITY)

    connections[f'ext2e'] = sim.Projection(
        pops[f'thalamus'], 
        pops[f'exc'], 
        ext_conn, 
        receptor_type='excitatory',
        synapse_type=sim.StaticSynapse(weight=0.1))

    connections[f'ext2i'] = sim.Projection(
        pops[f'thalamus'], 
        pops[f'inh'], 
        ext_conn, 
        receptor_type='excitatory',
        synapse_type=sim.StaticSynapse(weight=0.1))
    
    logger.info(f"Thalamic stimulus Done")
    return pops

if __name__ == '__main__':
    sim = get_sim()

    default_system_params = dict(n_neurons=200, 
            exc_conn_p=0.03, 
            inh_conn_p=0.02,
            synaptic_delay=2
            )
    
    default_lunchbox_params = dict(
                    timestep=0.1, 
                    time_scale_factor=50, # Defines the lunchbox where the systems will be runned on
                    duration=1000, 
                    min_delay=2,
                    # rng_seeds=[SEED],
                    neurons_per_core=250,
                    )
    sim_setup = {key:default_lunchbox_params[key] for key in ['timestep', 'min_delay', 'time_scale_factor']}
    sim.setup(**sim_setup)
    pops = build_system(default_system_params)
    print(sim_setup)
    sim.run(default_lunchbox_params['duration'])