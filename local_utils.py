"""Utility for getting the right simulator, 
DRY approach"""

import os
from rich import print as pprint

# Choice of the simulator is based on environment variables
simulator_name = os.environ.get("DJANLOO_NEURAL_SIMULATOR")

if simulator_name == "spiNNaker":
    pprint("choosing [blue]spiNNaker[/blue] as simulator")
    import pyNN.spiNNaker as sim
else:
    pprint("Choosing [green]neuron[/green] as simulator")
    import pyNN.neuron as sim
