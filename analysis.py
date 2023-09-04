import argparse
import os
import pickle
import logging

import matplotlib.pyplot as plt
from pyNN.utility.plotting import Figure, Panel

from local_utils import get_default_logger

parser = argparse.ArgumentParser(description='Analisys of simulation files')
parser.add_argument('folder', type=str, help='The folder of the simulation results')
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
Figure(
    # raster plot of the presynaptic neuron spike times
    Panel(results['population_exc', 'spikes'], xlabel="Time/ms", xticks=True,
          yticks=True, markersize=1, xlim=(0, 100)),
    title="Vogels-Abbott benchmark: excitatory cells spikes")
plt.tight_layout()
plt.show()
# # check results
# results = recover_results(outputs)
# results.keys()


# from pyNN.utility.plotting import Figure, Panel
# %matplotlib inline

# Figure(
#     # raster plot of the presynaptic neuron spike times
#     Panel(results['exc', 'spikes'], xlabel="Time/ms", xticks=True,
#           yticks=True, markersize=0.2, xlim=(0, tstop)),
#     title="Vogels-Abbott benchmark: excitatory cells spikes")

