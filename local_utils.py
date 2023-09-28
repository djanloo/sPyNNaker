"""Utilities for:

logging,
simulator choice,
minor computations
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from quantities import millisecond as ms

import logging
from rich.logging import RichHandler
from sklearn.linear_model import LinearRegression

_SIM_WAS_CHOSEN = False

def set_loggers(lvl=logging.DEBUG):
    
    # Sets rich to be the handler of logger
    rich_handler = RichHandler()
    rich_handler.setFormatter(logging.Formatter(fmt=f'[PID {os.getpid()}] %(message)s'))

    logging.basicConfig(format='%(message)s', 
                        handlers=[rich_handler])

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
        logger.debug(f"More than 10% of neurons were excluded from ISI CV computation ({excluded} over {n_neurons})")
        logger.debug(f"Average ISI CV is {np.mean(cvs)}")
    logger.debug(f"CVs are {cvs} (total of {len(cvs)}) (avg: {np.mean(cvs)})")
    return np.mean(cvs)


def random_subsample_synchronicity(block, binsize_ms=1, subsamp_size=20, n_samples=30, t_start=100, return_all=False):
    spiketrains = block.segments[0].spiketrains
    t_stop = spiketrains.t_stop
    t_start = t_start*ms
    binsize = binsize_ms * ms
    bins=np.linspace(t_start.magnitude, t_stop.magnitude, int((t_stop - t_start)/binsize)+1)
    samples_activities = np.zeros((n_samples, len(bins) -1))

    # Delete inactive neurons
    actives = []
    for sp in spiketrains:
        if len(sp) < 10:
            actives.append(sp)
    spiketrains = actives
    if len(actives) < subsamp_size:
        logger.warning(f"Could not estimate sync of net: active samples are less than subsample size")
        return np.nan
    
    indexes = np.arange(len(spiketrains))
    for sample_n in range(n_samples):
        np.random.shuffle(indexes)
        subsamp = []
        for subsamp_idx in indexes[:subsamp_size]:
            subsamp.append(spiketrains[subsamp_idx])

        subsamp_spikes = spiketrains_to_couples(subsamp)

        subsamp_spikes= subsamp_spikes[(subsamp_spikes[:, 1] > t_start.magnitude)&(subsamp_spikes[:, 1] < t_stop.magnitude)]
        # logger.info(f"Spiketimes of subsamp are {subsamp_spikes[:, 1]}")
        subsamp_act = np.histogram(subsamp_spikes[:, 1], bins=bins)[0]
        # logger.info(f"Subsamp avg activity is {subsamp_act}")
        samples_activities[sample_n] = subsamp_act
    if return_all:
        return samples_activities
    else:
        return np.mean(np.std(samples_activities, axis=0))


def chisync(block, 
            subsamp_sizes=[10, 20, 30, 40, 50, 60, 70],
            bootstrap_trials=3
            ):
    if avg_activity(block) < 2:
        logger.warning("chisync: population was too low in activity to compute chi. Returning 0.")
        return 0
    v = block.segments[0].filter(name="v")[0].magnitude.T
    indexes = np.arange(v.shape[0])

    syncs = np.zeros((len(subsamp_sizes), bootstrap_trials))
    for size_i in range(len(subsamp_sizes)):
        for bootstrap in range(bootstrap_trials):
            np.random.shuffle(indexes)
            subsamp_idxs = indexes[:subsamp_sizes[size_i]]
            
            sample_time_average = np.mean(v[subsamp_idxs], axis=0)
            # logger.debug(f"sample time-average has shape {sample_time_average.shape}")   

            variance_of_sample_mean = np.std(sample_time_average)**2
            # logger.debug(f"time-variance of time-average is {variance_of_sample_mean}")

            time_variance_of_single_neurons = np.std(v[subsamp_idxs], axis=1)**2
            # logger.debug(f"time variance of single neurons have shape {time_variance_of_single_neurons.shape}")

            mean_of_neural_variances = np.mean(time_variance_of_single_neurons)
            # logger.debug(f"mean_of_neural_variances is {mean_of_neural_variances}")

            syncs[size_i, bootstrap] = np.sqrt(variance_of_sample_mean/mean_of_neural_variances)
            # logger.debug(f"sync of bootstrap {i} is {syncs[i]}")
    
    ## Bootstrap averages
    syncs = np.mean(syncs, axis=-1)
    logger.debug(f"bootstrap averages of sync is {syncs}")

    ## Extrapolation
    model = LinearRegression()
    x = 1.0/np.sqrt(subsamp_sizes).reshape(-1,1)
    y = syncs.reshape(-1,1)

    model.fit(x,y)
    logger.debug(f"Linear model of sync returned chi_inf = {model.intercept_} and a = {model.coef_}")
    return model.intercept_[0]


def deltasync(block, subsamp_sizes=[10,20,30,40,50,60,70], bootstrap_trials=3, return_all=False):
    if avg_activity(block) < 2:
        logger.warning("deltasync: population was too low in activity to compute chi. Returning 0.")
        return 0
    v = block.segments[0].filter(name="v")[0].magnitude.T
    indexes = np.arange(v.shape[0])

    delta = np.zeros((len(subsamp_sizes), bootstrap_trials))
    
    for i in range(len(subsamp_sizes)):
        for bootstrap in range(bootstrap_trials):
            k = subsamp_sizes[i]
            np.random.shuffle(indexes)
            subsamp_idxs = indexes[:k]
            sample_average = np.mean(v[subsamp_idxs], axis=0) # average on neurons
            delta[i, bootstrap] = np.std(sample_average)**2 # variance over time

    # Average on bootstraps
    deltadelta = np.std(delta, axis=1)
    delta = np.mean(delta, axis=1)
    logger.debug(f"deltas are {delta} += {deltadelta}")

    ## Extrapolation
    model = LinearRegression()
    x = 1.0/np.array(subsamp_sizes).reshape(-1,1)
    y = delta.reshape(-1,1)

    model.fit(x,y)
    logger.debug(f"Linear model of sync returned chi_inf = {model.intercept_} and a = {model.coef_}")
    if return_all:
        return np.array(subsamp_sizes), delta, model.intercept_[0], model.coef_[0][0]
    
    return model.intercept_[0]

def active_fraction(block, n_spikes=10, t_start=50, t_end=None):
    spike_train_list = block.segments[0].spiketrains
    n_neurons = block.annotations['size']

    if t_end is None:
        t_end = spike_train_list.t_stop.magnitude

    active = 0
    for spikes in spike_train_list:
        spiketimes = spikes.magnitude[spikes.magnitude > t_start]
        if len(spiketimes) > n_spikes:
            active += 1

    return active/n_neurons

def rate_of_active_avg(block, n_spikes=10, t_start=50, t_end=None):
    spike_train_list = block.segments[0].spiketrains

    if t_end is None:
        t_end = spike_train_list.t_stop.magnitude

    fr = []
    for spikes in spike_train_list:
        spiketimes = spikes.magnitude[spikes.magnitude > t_start]
        if len(spiketimes) > n_spikes:
            fr += [len(spiketimes)/(t_end - t_start)*1e3]
    
    return np.mean(fr)


def rate_of_active_std(block, n_spikes=10, t_start=50, t_end=None):
    spike_train_list = block.segments[0].spiketrains

    if t_end is None:
        t_end = spike_train_list.t_stop.magnitude

    fr = []
    for spikes in spike_train_list:
        spiketimes = spikes.magnitude[spikes.magnitude > t_start]
        if len(spiketimes) > n_spikes:
            fr += [len(spiketimes)/(t_end - t_start)*1e3]
    
    return np.std(fr)
        

def isi_active_avg_tstd(block, t_start=50, t_end = None, n_spikes=10):

    spike_train_list = block.segments[0].spiketrains

    if t_end is None:
        t_end = spike_train_list.t_stop.magnitude

    active = 0.0
    isi_variance = 0.0
    for spikes in spike_train_list:
        spiketimes = spikes.magnitude[spikes.magnitude > t_start]
        if len(spiketimes) > n_spikes:
            active += 1
            isi_variance += np.std(np.diff(spiketimes))
    
    return isi_variance/active

def isi_active_avg_mean(block, t_start=50, t_end = None, n_spikes=10):
    spike_train_list = block.segments[0].spiketrains

    if t_end is None:
        t_end = spike_train_list.t_stop.magnitude

    active = 0.0
    isi_mean = 0.0
    for spikes in spike_train_list:
        spiketimes = spikes.magnitude[spikes.magnitude > t_start]
        if len(spiketimes) > n_spikes:
            active += 1
            isi_mean += np.mean(np.diff(spiketimes))
    
    return isi_mean/active


def v_regular_quants(block, n_quants=100, fraction=0.5, v_reset=-60.0, dv=0.1):
    """Assume that p(V, t) is independent from t, i.e. the system is in a stationary state"""
    assert fraction > 0 and fraction <=1, "Fraction must be between 0 and 1"

    v = block.segments[0].filter(name="v")[0].magnitude #shape = (time, neuron)

    # Takes only the last fraction
    v = v[int(fraction*len(v)):, :]

    # Removes divergent part
    v = v.reshape(-1)
    v = v[~((v > v_reset - dv)&(v < v_reset + dv))]
    quants = np.quantile(v, np.linspace(0,1, n_quants))

    return (quants,)

def v_divergent(block, fraction=0.5 ,v_reset=-60.0, dv=0.1):
    assert fraction > 0 and fraction <=1, "Fraction must be between 0 and 1"

    v = block.segments[0].filter(name="v")[0].magnitude #shape = (time, neuron)
    n_neurons = block.annotations['size']

    # Takes only the last fraction
    v = v[int(fraction*len(v)):, :]
    
    n_time_frames = v.shape[0]

    # Removes divergent part
    v = v.reshape(-1)
    return np.sum((v >= v_reset - dv)&( v <= v_reset + dv)) / n_time_frames / n_neurons
