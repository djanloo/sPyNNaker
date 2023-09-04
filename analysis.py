import argparse
import os
import pickle
import numpy as np

import matplotlib.pyplot as plt
from pyNN.utility.plotting import Figure, Panel

from local_utils import get_default_logger

parser = argparse.ArgumentParser(description='Analisys of simulation files')
parser.add_argument('folder', type=str, help='The folder of the simulation results')
parser.add_argument('--population', type=str, help='The population under analysis')

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

# Spikes
# Figure(
#     # raster plot of the presynaptic neuron spike times
#     Panel(results['population_exc', 'spikes'], xlabel="Time/ms", xticks=True,
#           yticks=True, markersize=1, xlim=(0, 100)),
#     title="Vogels-Abbott benchmark: excitatory cells spikes")

# V-density
signal = results[args.population, "v"]
print(signal.shape)
nbins = 40
hist=np.zeros((nbins, len(signal)))

v_bins = np.linspace(np.min(signal), np.max(signal), nbins+1)
X, Y = np.linspace(0,len(signal), len(signal)), v_bins[:-1]
X, Y = np.meshgrid(X,Y)

for time_index in range(len(signal)):
    hist[:, time_index] = np.histogram(signal[time_index], bins=v_bins, density=True)[0]*100

plt.contourf(X, Y, hist)
plt.colorbar()
plt.xlabel("time [ms]")
plt.ylabel("V [mV]")
plt.title(rf"$\rho(V, t)$ for {args.population}")
plt.tight_layout()

# On the remote server save instead of showing
if os.environ.get("USER") == "bbpnrsoa":
    plt.savefig(f"{folder_name}/analysis.png")
else:
    plt.show()