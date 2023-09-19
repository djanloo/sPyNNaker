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
from local_utils import random_subsample_synchronicity

lunchbox = LunchBox.from_folder("n_600_conn_scan")
logger=logging.getLogger("APPLICATION")

def sync(block):
    return random_subsample_synchronicity(block, n_samples=20, subsamp_size=20)

lunchbox.add_extraction(sync)
lunchbox._extract()


sincronicity_results = lunchbox.get_extraction_triplets("exc_conn_p", "inh_conn_p", "sync")
data = sincronicity_results['exc']['sync']




triang = tri.Triangulation(*(sincronicity_results['exc']['exc_conn_p', 'inh_conn_p'].T))
mappable=plt.tricontourf(triang, data)
plt.scatter(*(sincronicity_results['exc']['exc_conn_p', 'inh_conn_p'].T), 
            c=sincronicity_results['exc']['sync'], edgecolor="k", marker="s")
plt.colorbar(mappable)



plt.xlabel("Excitatory connectivity")
plt.ylabel("Inhibitory connectivity")
# sys_1 = 140338675003552 #low
# sys_2 = 139992135052320

# for sys in [sys_1, sys_2]:
#     syn = sync(lunchbox.systems[sys].pops['exc'])
#     plt.plot(np.mean(syn, axis=1))
plt.show()