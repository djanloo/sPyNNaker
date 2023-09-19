"""Postprocessing on runs"""
import numpy as np
import logging
from quantities import millisecond as ms
from matplotlib import tri
from matplotlib import pyplot as plt
from elephant.spike_train_correlation import spike_time_tiling_coefficient
from elephant.conversion import BinnedSpikeTrain
from run_manager import LunchBox
from local_utils import spiketrains_to_couples, avg_activity
from local_utils import set_loggers; set_loggers()

lunchbox = LunchBox.from_folder("n_600_conn_scan")
logger=logging.getLogger("APPLICATION")

def sincronicity(block):
    # avg_act = avg_activity(block)
    # bin_size = 0.1/avg_act*ms
    spike_array = spiketrains_to_couples(block.segments[0].spiketrains)
    # duration = block.segments[0].spiketrains.t_stop
    # logger.debug(f"bin_size is {bin_size} ({int(duration/bin_size)} bins)")

    activity = np.histogram(spike_array.T[1], bins=300, density=True)[0]
    
    autocorrelation = np.convolve(activity, activity[::-1], mode='full') / (np.linalg.norm(activity) ** 2)
    return np.sum(autocorrelation)

lunchbox.add_extraction(sincronicity)
lunchbox._extract()


sincronicity_results = lunchbox.get_extraction_triplets("exc_conn_p", "inh_conn_p", "sincronicity")
data = sincronicity_results['exc']['sincronicity']

logger.info(f"sync is:\n{lunchbox.extractions}")

triang = tri.Triangulation(*(sincronicity_results['exc']['exc_conn_p', 'inh_conn_p'].T))

# Plot bad points as "X"
plt.scatter(*(sincronicity_results['exc']['exc_conn_p', 'inh_conn_p'].T), 
        color='red', edgecolor="k", marker="X", zorder =10)

# Plot the contourf
mappable=plt.tricontourf(triang, data)

# Plot the good points as squares
plt.scatter(*(sincronicity_results['exc']['exc_conn_p', 'inh_conn_p'].T), 
            c=sincronicity_results['exc']['sincronicity'], edgecolor="k", marker="s")

plt.colorbar(mappable)



plt.xlabel("Excitatory connectivity")
plt.ylabel("Inhibitory connectivity")


# plt.figure(2)
# sys_1 = 140338675003552 #low
# sys_2 = 139992135052320

# bins = [300, 300]

# for sys, b in zip([sys_1, sys_2], bins):
#     spike_array = spiketrains_to_couples(lunchbox.systems[sys].pops['exc'].segments[0].spiketrains)
  
#     activity = np.histogram(spike_array.T[1], bins=b, density=False)[0]
#     plt.step(np.linspace(0,1000, b), activity, label=f"sync = {sincronicity(lunchbox.systems[sys].pops['exc'])}")
# plt.legend()

# fig, axes = plt.subplots(2,1, sharex=True)

# for sys, b, ax in zip([sys_1, sys_2], bins, axes):
#     spike_array = spiketrains_to_couples(lunchbox.systems[sys].pops['exc'].segments[0].spiketrains)
  
#     ax.scatter(*(spike_array.T), label=f"sync = {sincronicity(lunchbox.systems[sys].pops['exc'])}", s=1, alpha=0.8)
#     ax.legend()

plt.show()