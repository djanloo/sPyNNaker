"""Utility for getting the right simulator, 
DRY approach"""

import os
import logging
from rich import print as pprint
import matplotlib.pyplot as plt
import numpy as np

def get_sim():
    # Choice of the simulator is based on environment variables
    simulator_name = os.environ.get("DJANLOO_NEURAL_SIMULATOR")

    if simulator_name == "spiNNaker":
        pprint("choosing [blue]spiNNaker[/blue] as simulator")
        import pyNN.spiNNaker as sim
    elif simulator_name == 'neuron':
        pprint("Choosing [green]neuron[/green] as simulator")
        import pyNN.neuron as sim
    elif simulator_name == 'brian':
        pprint("Choosing [yellow]brian2[/yellow] as simulator")
        import pyNN.brian2 as sim
    elif simulator_name == 'nest':
        pprint("Choosing [magenta]nest[/magenta] as simulator")
        import pyNN.nest as sim
    else:
        raise ValueError(f"Simulator defined in the environment [{simulator_name}] is not implemented")
        pprint("Simulator is not specified by [red]DJANLOO_NEURAL_SIMULATOR[/red]\nDefaulting to [green]neuron[/green]...")
        import pyNN.neuron as sim

    return sim

def get_default_logger(logger_name, lvl=logging.DEBUG):
    logger = logging.getLogger(logger_name)

    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s-%(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    return logger

logger = get_default_logger("UTILIITIES")
def annotate_dict(dict_, ax):
    ys = np.linspace(0,1, len(dict_)+2)
    for key, y in zip(dict_.keys(), ys[1:-1]):
        ax.annotate(f"{key}={dict_[key]}", (0, y), ha='center')
    
    ax.set_xlim(-1,1)
    ax.set_ylim(0,1)
    ax.axis('off')

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
    logger.debug(f"Spike arrays have shape {spike_array.shape}")
    logger.debug(f"Spike arrays are {spike_array}")
    logger.debug(f"Spike arrays.T are {spike_array.T}")
    return spike_array

def avg_activity(n_neurons, spike_train_list, t_start=50, t_end=None):
    if t_end is None:
        t_end = spike_train_list.t_stop.magnitude

    spike_array = spiketrains_to_couples(spike_train_list)

    # Selects only the ones after burn-in
    spike_array= spike_array[(spike_array[:, 1] > t_start)&(spike_array[:, 1] < t_end)]
    logger.debug(f"number of spikes in interval [{t_start},{t_end}] ms: \t\t\t{len(spike_array):10}")
    logger.debug(f"average number of spikes in interval [{t_start},{t_end}] ms: \t\t{len(spike_array)/(t_end - t_start):10.2f} spikes/ms")
    logger.debug(f"average activity per neuron in interval [{t_start},{t_end}] ms: \t{len(spike_array)/(t_end - t_start)/n_neurons*1e3:10.2f} spikes/sec")
    return len(spike_array)/(t_end - t_start)/n_neurons

def avg_current(V, ge, gi):
    return np.mean(V*(ge - gi), axis=0)
