"""Utilities for:

logging,
simulator choice,
minor computations
"""

import os
import matplotlib.pyplot as plt
import numpy as np

import logging
from rich.logging import RichHandler


_SIM_WAS_CHOSEN = False

def set_loggers(lvl=logging.DEBUG):
    
    # Sets rich to be the handler of logger
    rich_handler = RichHandler()
    rich_handler.setFormatter(logging.Formatter(fmt='%(message)s'))

    logging.basicConfig(format='%(message)s', 
                        handlers=[rich_handler])

    # For each logger of the submodules sets the verbosity
    for logger_name in ["ANALYSIS", "PLOTTING", "UTILS", "APPLICATION", "RUN_MANAGER",  "NETWORK_BUILDING"]:
        _ = logging.getLogger(logger_name)
        
        _.setLevel(lvl)

set_loggers()
logger = logging.getLogger("UTILS")


def get_sim():
    # Choice of the simulator is based on environment variables
    simulator_name = os.environ.get("DJANLOO_NEURAL_SIMULATOR")

    if simulator_name == "spiNNaker":
        if not _SIM_WAS_CHOSEN:
            logger.info("choosing [blue]SPINNAKER[/blue] as simulator",
                        extra={"markup": True})
        import pyNN.spiNNaker as sim
    elif simulator_name == 'neuron':
        if not _SIM_WAS_CHOSEN:
            logger.info("Choosing [green]NEURON[/green] as simulator",
                        extra={"markup": True})
        import pyNN.neuron as sim
    elif simulator_name == 'brian':
        if not _SIM_WAS_CHOSEN:
            logger.info("Choosing [yellow]BRIAN2[/yellow] as simulator",
                        extra={"markup": True})
        import pyNN.brian2 as sim
    elif simulator_name == 'nest':
        if not _SIM_WAS_CHOSEN:
            logger.info("Choosing [magenta]NEST[/magenta] as simulator",
                        extra={"markup": True})
        import pyNN.nest as sim
    else:
        raise ValueError(f"Simulator defined in the environment [{simulator_name}] is not implemented")
        import pyNN.neuron as sim
    _SIM_WAS_CHOSEN = True
    return sim


def activity_labeller(spiketrainlist):

    time_window_start = spiketrainlist.t_start + (spiketrainlist.t_stop - spiketrainlist.t_start)*9/10
    time_window_stop = spiketrainlist.t_stop

    raise NotImplementedError("TODO")

def spiketrains_to_couples(spike_train_list):
    # La SpikeTrainList in un formato [neurone, tempo di spike]
    spike_array = []

    for neuron_index, spike_train in enumerate(spike_train_list):

        # Generates an array of [n_idx] that is len(spikelist) long
        neuron_indices = (np.ones(len(spike_train))*neuron_index).astype(int)
        # Times to numpy
        spike_times = spike_train.times.magnitude
        spike_array.append(np.column_stack((neuron_indices, spike_times)))

    spike_array = np.vstack(spike_array)
    # logger.debug(f"Spike arrays have shape {spike_array.shape}")
    # logger.debug(f"Spike arrays are \n{spike_array}")
    # logger.debug(f"Spike arrays.T are \n{spike_array.T}")
    return spike_array

def avg_activity(block, t_start=50, t_end=None):

    spike_train_list = block.segments[0].spiketrains
    n_neurons = block.annotations['size']
    if t_end is None:
        t_end = spike_train_list.t_stop.magnitude

    spike_array = spiketrains_to_couples(spike_train_list)

    # Selects only the ones after burn-in
    spike_array= spike_array[(spike_array[:, 1] > t_start)&(spike_array[:, 1] < t_end)]
    # logger.info(f"number of spikes in interval [{t_start},{t_end}] ms: \t\t\t{len(spike_array):10}")
    # logger.info(f"average number of spikes in interval [{t_start},{t_end}] ms: \t\t{len(spike_array)/(t_end - t_start):10.2f} spikes/ms")
    logger.info(f"average activity per neuron in interval [{t_start},{t_end}] ms: \t{len(spike_array)/(t_end - t_start)/n_neurons*1e3:10.2f} spikes/sec")
    return len(spike_array)/(t_end - t_start)/n_neurons*1e3

# def avg_activity_by_spiketrains(n_neurons, spike_train_list, t_start=50, t_end=None):

#     # spike_train_list = population.getSpikes().segments[0].spiketrains
#     # n_neurons = population.get_data().annotations['size']
#     if t_end is None:
#         t_end = spike_train_list.t_stop.magnitude

#     spike_array = spiketrains_to_couples(spike_train_list)

#     # Selects only the ones after burn-in
#     spike_array= spike_array[(spike_array[:, 1] > t_start)&(spike_array[:, 1] < t_end)]
#     logger.info(f"number of spikes in interval [{t_start},{t_end}] ms: \t\t\t{len(spike_array):10}")
#     logger.info(f"average number of spikes in interval [{t_start},{t_end}] ms: \t\t{len(spike_array)/(t_end - t_start):10.2f} spikes/ms")
#     logger.info(f"average activity per neuron in interval [{t_start},{t_end}] ms: \t{len(spike_array)/(t_end - t_start)/n_neurons*1e3:10.2f} spikes/sec")
#     return len(spike_array)/(t_end - t_start)/n_neurons*1e3

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)
    

def avg_isi_cv(block, t_start=100, t_end=None):

    spike_train_list = block.segments[0].spiketrains
    n_neurons = block.annotations['size']
    
    if t_end is None:
        t_end = spike_train_list.t_stop.magnitude

    if t_end is None:
        t_end = spike_train_list.t_stop.magnitude

    cvs = []
    excluded = 0
    for neuron_idx, spikes in enumerate(spike_train_list):
        spiketimes = spikes.magnitude[spikes.magnitude > t_start]
        if len(spiketimes) < 10:
            excluded += 1
        else:
            isi = np.diff(spiketimes)
            current_cv = np.std(isi)/np.mean(isi)
            if np.isnan(current_cv):
                logger.warning(f"CV of neuron {neuron_idx} turned out to be NaN:\nspikes={spiketimes}\nisi={isi}")
            cvs += [current_cv]
    if excluded > 0.1*n_neurons:
        logger.warning(f"More than 10% of neurons were excluded from ISI CV computation ({excluded} over {n_neurons})")
        logger.warning(f"Average ISI CV is {np.mean(cvs)}")
    logger.debug(f"CVs are {cvs} (total of {len(cvs)}) (avg: {np.mean(cvs)})")
    return np.mean(cvs)
