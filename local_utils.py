"""Utility for getting the right simulator, 
DRY approach"""

import os
from rich import print as pprint

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
