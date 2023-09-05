import argparse
import os
import pickle
import numpy as np

import matplotlib.pyplot as plt
from pyNN.utility.plotting import Figure, Panel

from local_utils import get_default_logger
import seaborn as sns

possible_plots = ["quantiles_v", "density_v", "spikes"]
parser = argparse.ArgumentParser(description='Analisys of simulation files')
parser.add_argument('folder', type=str, help='The folder of the simulation results')
parser.add_argument('--population', type=str, help='The population under analysis')
parser.add_argument('--plot', type=str, choices=possible_plots, nargs='+', default="spikes",
                            help=f"Plot t be displayed, choices are: {','.join(possible_plots)}")

args = parser.parse_args()
folder_name = args.folder

logger = get_default_logger("analysis")

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
analog_fig, analog_ax = plt.subplots(figsize=(6,5))
spike_fig, spike_axes = plt.subplot_mosaic([["spikes", "neuron_activity"], ["time_activity", "none"]],
                                                        height_ratios=[1,0.5],
                                                        width_ratios=[1, 0.5],
                                                        figsize=(6,5), 
                                                        sharex=False, 
                                                        sharey=False,
                                                        constrained_layout=True)

# Spikes & activity
if "spikes" in args.plot:
    spikelist = results[args.population, 'spikes']

    data = []
    for neuron_index, neuron_spikes in enumerate(spikelist):
        for neuron_spike_time in neuron_spikes.times:
            data.append([neuron_spike_time, neuron_index])

    data = np.array(data)
    spike_axes['spikes'].scatter(*(data.T), marker=".", color="k")
    spike_axes['spikes'].set_xlim((spikelist.t_start, spikelist.t_stop))

    # Activity in time
    act_t = np.histogram(data.T[0], bins=np.linspace(spikelist.t_start, spikelist.t_stop, 42 +1), density=True)[0]
    print(act_t.shape)
    spike_axes['time_activity'].step(np.linspace(spikelist.t_start, spikelist.t_stop, 42), act_t)

    # Activity in neuron
    print(data.T[1])
    _, act_n = np.unique(data.T[1], return_counts=True)
    act_n = np.concatenate( (act_n, [0]*(2000 - len(act_n))))

    print(act_n, len(act_n))
    _, act_n = np.unique(act_n, return_counts=True)
    print(act_n, len(act_n))
    spike_axes['neuron_activity'].barh(range(len(act_n)), act_n )
    spike_axes['neuron_activity'].set_xscale("log")
    spike_axes['none'].axis("off")


# V-density
if "density_v" in args.plot:
    signal = results[args.population, "v"]

    nbins = 40
    hist=np.zeros((nbins, len(signal)))

    v_bins = np.linspace(np.min(signal), np.max(signal), nbins+1)
    X, Y = np.linspace(0,len(signal), len(signal)), v_bins[:-1]
    X, Y = np.meshgrid(X,Y)

    for time_index in range(len(signal)):
        hist[:, time_index] = np.histogram(signal[time_index], bins=v_bins, density=True)[0]*100
    levels = [0,1,2,3,4,5,10,15,20, 25]

    cbar = analog_fig.colorbar(
                                analog_ax.contourf(X, Y, hist, levels=levels)
                            )
    
    cbar.set_label('numerical density [%]', rotation=270, size=10)
    cbar.ax.set_yticks(levels)
    analog_ax.set_xlabel("time [ms]", size=10)
    analog_ax.set_ylabel("V [mV]", size=10)
    analog_ax.set_title(rf"$\rho(V, t)$ for {args.population} ({os.environ.get('DJANLOO_NEURAL_SIMULATOR')})", size=13)
    analog_ax.set_ylim(-75, -45)

# Quantiles
if "quantiles_v" in args.plot:
    qq = np.quantile(np.array(signal), [.1,.2,.3,.4, .5, .6, .7, .8, .9], axis=1)
    colors = sns.color_palette("Accent", n_colors=qq.shape[0])

    for q,c,l in zip(qq, colors, range(1,10)):
        analog_ax.plot(q, color=c,label=f"{l*10}-percentile")
    analog_ax.legend(ncols=3, fontsize=8)
    analog_ax.set_title("quantiles of V(t)")
    analog_ax.set_xlabel("t [ms]")
    analog_ax.set_ylabel("V [mV]")

# On the remote server save instead of showing
if os.environ.get("USER") == "bbpnrsoa":
    plt.savefig(f"{folder_name}/analysis.png")
else:
    plt.show()