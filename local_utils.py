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
    else:
        pprint("Simulator is not specified by [red]DJANLOO_NEURAL_SIMULATOR[/red]\nDefaulting to [green]neuron[/green]...")
        import pyNN.neuron as sim

    return sim

def get_default_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    return logger

def annotate_dict(dict_, ax):
    ys = np.linspace(0,1, len(dict_)+2)
    for key, y in zip(dict_.keys(), ys[1:-1]):
        ax.annotate(f"{key}={dict_[key]}", (0, y), ha='center')
