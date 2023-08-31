"""Small test for SpikeSourcePoisson
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pyNN.neuron as sim
from pyNN.utility import SimulationProgressBar
from pyNN.utility.plotting import plot_spiketrains

from time import perf_counter
from rich import print as pprint

# Choice of the simulator is based on environment variables
import os
simulator_name = os.environ.get("DJANLOO_NEURAL_SIMULATOR")

if simulator_name == "spiNNaker":
    pprint("choosing [blue]spiNNaker[/blue] as simulator")
    import pyNN.spiNNaker as sim
else:
    pprint("Choosing [green]neuron[/green] as simulator")
    import pyNN.neuron as sim


simulation_time = 50
excitatory_strength = 0.05
inhibitory_strength = 1.0
connection_probability = 0.02

dt = 1
N = 100

sim.setup(timestep=dt)
print("setup complete")

# Neurons
bombing_population = sim.Population(1, sim.SpikeSourcePoisson(rate=300))
target_population = sim.Population(N, sim.IF_cond_exp())
calming_population = sim.Population(N//4, sim.IF_cond_exp())

print("population complete")

# Connections
bombing_connections = sim.Projection(bombing_population, target_population, 
                                sim.AllToAllConnector(), 
                                sim.StaticSynapse(weight=excitatory_strength, 
                                                  delay=1.0), 
                                receptor_type='excitatory')
print("connections 1/4")
self_connections = sim.Projection(target_population, target_population,
                                  sim.FixedProbabilityConnector(connection_probability),
                                  sim.StaticSynapse(weight=excitatory_strength/N/connection_probability, 
                                                    delay=1.0))
print("connections 2/4")
calming_connections = sim.Projection(calming_population, target_population,
                                     sim.AllToAllConnector(),
                                     sim.StaticSynapse(weight=inhibitory_strength/N, 
                                                       delay=2.0), 
                                     receptor_type="inhibitory")
print("connections 3/4")
calm_stimulating = sim.Projection(   target_population, calming_population,
                                     sim.AllToAllConnector(),
                                     sim.StaticSynapse(weight=excitatory_strength/N, delay=1.0))
print("connections done")

# Define recording variables
target_population.record(("v", "spikes"))

# Starts the simulation
start = perf_counter()
print("starting")
sim.run(simulation_time, callbacks=[SimulationProgressBar(100*dt, simulation_time)])
runtime = perf_counter() - start
print(f"run time: {runtime}")

# Plots the data
block = target_population.get_data()
print(f"{len(block.segments)} segments of block found")


fig, (axspike, axv) = plt.subplots(2,1, sharex=True)
for signal in block.segments[0].analogsignals:
    print(f"data is {signal.times}")
    print(f"data has length {len(signal.times)}")
    axv.plot(signal.times, signal.magnitude)

plot_spiketrains(axspike, block.segments[0].spiketrains )
axspike.set_title(f"Runtime: {runtime:.3f} s")
fig.savefig(f"results_{sim.__name__}.png")