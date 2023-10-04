"""Utilities for:

logging,
simulator choice,
minor computations
"""

import os
import numpy as np
from quantities import millisecond as ms

import logging
from rich.logging import RichHandler
from sklearn.preprocessing import StandardScaler
from scipy.signal import decimate

_SIM_WAS_CHOSEN = False

def set_loggers(lvl=logging.DEBUG):
    
    # Sets rich to be the handler of logger
    rich_handler = RichHandler()
    rich_handler.setFormatter(logging.Formatter(fmt=f'[PID {os.getpid()}] %(message)s'))

    file_handler = logging.FileHandler(f"LOGS/logs_{os.getpid()}.txt")
    file_handler.setFormatter(logging.Formatter(fmt=f'[PID {os.getpid()}] %(message)s'))
    file_handler.setLevel(logging.DEBUG)

    logging.basicConfig(format='%(message)s', 
                        handlers=[rich_handler, file_handler],
                        )

    # For each logger of the submodules sets the verbosity
    for logger_name in ["ANALYSIS", "PLOTTING", "UTILS", "APPLICATION", "RUN_MANAGER",  "NETWORK_BUILDING"]:
        _ = logging.getLogger(logger_name)
        
        _.setLevel(lvl)

def set_logger_pid(logger):
    logger = logging.getLogger()
    logger.handlers[0].setFormatter(logging.Formatter(fmt=f'[PID {os.getpid()}] %(message)s'))


set_loggers()
logger = logging.getLogger("UTILS")


def get_sim():
    global _SIM_WAS_CHOSEN
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

    return spike_array

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)
    

def activity_stats(block, n_spikes=10, t_start=50, t_end=None):
    spike_train_list = block.segments[0].spiketrains
    n_neurons = block.annotations['size']

    if t_end is None:
        t_end = spike_train_list.t_stop.magnitude

    active_firing_rates = []
    all_firing_rates = []

    for spikes in spike_train_list:
        spiketimes = spikes.magnitude[spikes.magnitude > t_start]
        all_firing_rates += [len(spiketimes)/(t_end - t_start)*1e3]

        if len(spiketimes) > n_spikes:
            active_firing_rates += [len(spiketimes)/(t_end - t_start)*1e3]

    # Return results
    stats = dict( active_fraction=len(active_firing_rates)/n_neurons)
    stats['avg_activity'] = np.mean(all_firing_rates) if len(all_firing_rates) > 0 else np.nan
    stats['rate_of_active_avg']=np.mean(active_firing_rates) if len(active_firing_rates) > 0 else np.nan
    stats['rate_of_active_std']=np.std(active_firing_rates) if len(active_firing_rates) > 0 else np.nan

    return stats
    

def isi_stats(block, t_start=50, t_end = None, n_spikes=10):
    spike_train_list = block.segments[0].spiketrains

    if t_end is None:
        t_end = spike_train_list.t_stop.magnitude

    active = 0.0
    isi_mean = 0.0
    isi_variance = 0.0
    isi_cv = 0.0

    for spikes in spike_train_list:
        spiketimes = spikes.magnitude[spikes.magnitude > t_start]
        if len(spiketimes) > n_spikes:
            active += 1
            isi_mean += np.mean(np.diff(spiketimes))
            isi_variance += np.std(np.diff(spiketimes))
            isi_cv += isi_variance/isi_mean

    # Return results
    stats = dict()
    stats['isi_mean_avg'] = isi_mean/active if active > 0 else np.nan
    stats['isi_tstd_avg'] = isi_variance/active if active > 0 else np.nan
    stats['isi_cv_avg'] = isi_cv/active if active > 0 else np.nan
    
    return stats


def v_stats(block, n_bins=50, fraction=0.5, v_reset=-61, dv=0.1):

    v = block.segments[0].filter(name="v")[0].magnitude #shape = (time, neuron)
    n_neurons = block.annotations['size']

    # Takes only the last fraction
    v = v[-int(fraction*len(v)):, :]
    
    n_time_frames = v.shape[0]

    # Removes divergent part
    v = v.reshape(-1)
    v_regular = v[~((v > v_reset - dv)&(v < v_reset + dv))]
    is_v_divergent = ((v > v_reset - dv)&(v < v_reset + dv))

    bins_string = ','.join([str(val) for val in np.histogram(v_regular, n_bins, density=True)[0]])

    stats = dict(v_regular_binned=bins_string, 
                 v_divergent=np.sum(is_v_divergent)/n_time_frames/n_neurons)
    return stats

def synaptic_conductance_stats(block, fraction = 0.1):

    # Extract time delta of integration and timestep
    time = (block.segments[0].spiketrains.t_stop - block.segments[0].spiketrains.t_start).magnitude # ms
    n_time_points = block.segments[0].filter(name="v")[0].magnitude.shape[0]

    Delta_t = fraction*time
    dt = Delta_t/(fraction*n_time_points)

    stats = dict()
    for quantity in ["gsyn_exc", "gsyn_inh"]:

        q_t = block.segments[0].filter(name=quantity)[0].magnitude #shape = (time, neuron)
        q_t = q_t[int(fraction*n_time_points):, :]

        # Ensemble average
        q_t_avg = np.mean(q_t, axis=1)

        # Time average
        q_avg_mean = np.trapz(q_t_avg, dx=dt)/Delta_t
        stats[f"{quantity}_avg_mean"] = q_avg_mean

    return stats

def phase_invariant_average(block, fraction=0.1):
    def align_by_best_match(signal, reference_signal, tol=0.2):
        scaler = StandardScaler()
        signal_rescaled = scaler.fit_transform(signal.reshape(-1,1)).reshape(-1)
        reference_signal_rescaled = scaler.fit_transform(reference_signal.reshape(-1,1)).reshape(-1)
        dists = np.zeros(len(reference_signal_rescaled))

        # Decimates the signal to speed up the computation
        # Assuming a maximum spiking frequency of 200 Hz
        # I take roughly 5 points per period
        # So i decimate the signal until is 128 points long

        signal_downsampled = signal_rescaled.copy()
        reference_signal_downsampled = reference_signal_rescaled.copy()
        time_scaling = 1

        while len(signal_downsampled) > 128:
            signal_downsampled = decimate(signal_downsampled, 2)
            reference_signal_downsampled = decimate(reference_signal_downsampled, 2)
            time_scaling  *= 2 

        for i in range(1, len(signal_downsampled)):
            rounded_signal = signal_downsampled.copy()
            rounded_signal[:i] = signal_downsampled[-i:]
            rounded_signal[i:] = signal_downsampled[:-i]
            dists[i] = np.sum((rounded_signal - reference_signal_downsampled)**2)
            if i > 1 and dists[i] < dists[i-1] and dists[i] < tol*len(signal_downsampled):
                break
        
        best_match = np.argmin(dists[1:]) + 1
        best_match_signal = signal.copy()
        best_match_signal[:best_match] = signal[-time_scaling*best_match:]
        best_match_signal[best_match:] = signal[:-time_scaling*best_match]

        return best_match_signal
    
    v = block.segments[0].filter(name="v")[0].magnitude.T #shape = (neuron, time)

    # Gets only last fraction
    v = v[:, -int(fraction*v.shape[1]):]

    aligned_v = np.zeros(v.shape)
    for i in range(1,len(aligned_v)):
        aligned_v[i] = align_by_best_match(v[i], v[0])

    strvalues = [f"{v:.2f}" for v in np.mean(aligned_v, axis=0)]

    return ','.join(strvalues)

