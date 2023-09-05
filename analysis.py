import argparse
import os
import pickle
import numpy as np

import matplotlib.pyplot as plt
plt.style.use("./style.mplstyle")
from pyNN.utility.plotting import Figure, Panel

from local_utils import get_default_logger
import seaborn as sns

logger = get_default_logger("analysis")

possible_plots = ["quantiles", "density", "spikes"]
quantities = ["v", "gsyn_exc", "gsyn_inh"]

parser = argparse.ArgumentParser(description='Analisys of simulation files')

parser.add_argument('folder', 
                    type=str, 
                    help='the folder of the simulation results')

parser.add_argument('population', 
                    type=str, 
                    help='the population under analysis')

parser.add_argument('--plot', 
                    type=str, 
                    choices=possible_plots, 
                    nargs='+', 
                    default="spikes",
                    help=f"plot to be displayed, choices are: {', '.join(possible_plots)}")

parser.add_argument('--bins', 
                    type=int,
                    default=30,
                    help="the number of bins for activity plot"
                    )
parser.add_argument('--quantity',
                   type=str,
                   default='v',
                   choices=quantities,
                   help=f"the quantity to plot, chosed among: {', '.join(quantities)}"
                   )
parser.add_argument('--v', 
                    type=int, 
                    default=10,
                    help="verbosity level")

args = parser.parse_args()

logger.setLevel(args.v)

folder_name = args.folder
files = [f for f in os.listdir(folder_name) if f.endswith(".pkl")]
results = dict()

for file in files:
    neo_blocks = []
    with (open(f"{folder_name}/{file}", "rb")) as openfile:
        while True:
            try:
                neo_blocks.append(pickle.load(openfile))
            except EOFError:
                break

    if len(neo_blocks) > 1:
        logger.error(f"more than one neo blocks were found in file {file}")
        exit()

    # Analog signals
    for analogsignal in neo_blocks[0].segments[0].analogsignals:
        results[file.replace(".pkl", ""), analogsignal.name] = analogsignal
        logger.info(f"In file: {file.replace('.pkl', ''):20} found signal: {analogsignal.name:20}")

    # Spike trains
    results[file.replace(".pkl", ""), "spikes"] = neo_blocks[0].segments[0].spiketrains
    logger.info(f"In file: {file.replace('.pkl', ''):20} found signal: spikes")

########## PLOTTING #########
analog_fig, analog_ax, spike_fig, spike_axes = None, None, None, None

# Spikes & activity
if "spikes" in args.plot:
    spike_fig, spike_axes = plt.subplot_mosaic([["spikes", "neuron_activity"], ["time_activity", "none"]],
                                                        height_ratios=[1,0.5],
                                                        width_ratios=[1, 0.5],
                                                        figsize=(6,5), 
                                                        sharex=False, 
                                                        sharey=False,
                                                        constrained_layout=True)
    spikelist = results[args.population, 'spikes']
    data = []
    for neuron_index, neuron_spikes in enumerate(spikelist):
        for neuron_spike_time in neuron_spikes.times:
            data.append([neuron_spike_time, neuron_index])
    data = np.array(data)

    # Spikes
    spike_axes['spikes'].scatter(*(data.T), marker=".", color="k")
    spike_axes['spikes'].set_xlim((spikelist.t_start, spikelist.t_stop))

    # Activity in time
    act_t = np.histogram(data.T[0], bins=np.linspace(spikelist.t_start, spikelist.t_stop, 42 +1), density=True)[0]
    spike_axes['time_activity'].step(np.linspace(spikelist.t_start, spikelist.t_stop, 42), act_t)
    # Details
    spike_axes['time_activity'].set_xlabel("t [ms]")
    spike_axes['time_activity'].set_ylabel("PSTH")

    # Activity in neuron
    _, act_n = np.unique(data.T[1], return_counts=True)
    act_n = np.concatenate( (act_n, [0]*(2000 - len(act_n))))
    _, act_n = np.unique(act_n, return_counts=True)
    spike_axes['neuron_activity'].barh(range(len(act_n)), act_n)
    spike_axes['neuron_activity'].set_xscale("log")
    # Details
    spike_axes['neuron_activity'].set_xlabel("# of neurons")
    spike_axes['neuron_activity'].set_ylabel("# of activations")

    # Turn off the dummy corner plots
    spike_axes['none'].axis("off")


# V-density
if "density" in args.plot:
    analog_fig, analog_ax = plt.subplots(figsize=(6,5))
    analog_ax.set_xlabel("t [ms]")
    analog_ax.set_ylabel("V [mV]")

    signal = results[args.population, args.quantity]
    logger.debug(f"signal has shape {signal.shape}")

    hist=np.zeros((args.bins, len(signal)))

    v_bins = np.linspace(np.min(signal), np.max(signal), args.bins+1)
    X, Y = np.linspace(0,len(signal), len(signal)), v_bins[:-1]
    X, Y = np.meshgrid(X,Y)

    for time_index in range(len(signal)):
        hist[:, time_index] = np.log10(np.histogram(signal[time_index], bins=v_bins, density=True)[0])

    hist[~np.isfinite(hist)] = np.min(hist[np.isfinite(hist)])
    logger.info(f"-inf valued areas in log density have been replaced with minimal {np.min(hist[np.isfinite(hist)])}")
    cbar = analog_fig.colorbar(
                                analog_ax.contourf(X, Y, hist, levels=10)
                            )
    cbar.set_label('log density', rotation=270, size=10)
    
    analog_ax.set_xlabel("t [ms]")
    analog_ax.set_ylabel("V [mV]")

# Quantiles
if "quantiles" in args.plot:
    if analog_fig is None:
        analog_fig, analog_ax = plt.subplots(figsize=(6,5))
        analog_ax.set_xlabel("t [ms]")
        analog_ax.set_ylabel("V [mV]")

    signal = results[args.population, args.quantity]
    qq = np.quantile(np.array(signal), [.1,.2,.3,.4, .5, .6, .7, .8, .9], axis=1)
    colors = sns.color_palette("Accent", n_colors=qq.shape[0])

    for q,c,l in zip(qq, colors, range(1,10)):
        analog_ax.plot(q, color=c,label=f"{l*10}-percentile")
    analog_ax.legend(ncols=3, fontsize=8)
    analog_ax.set_title("quantiles of V(t)")
    
    analog_fig.show()

# On the remote server save instead of showing
if os.environ.get("USER") == "bbpnrsoa":
    analog_fig.savefig(f"{folder_name}/analysis_analog.png")
    spike_fig.savefig(f"{folder_name}/analysis_spikes.png")
else:
    plt.show()
    pass