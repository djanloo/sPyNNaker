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
from local_utils import random_subsample_synchronicity, potential_random_subsample_synchronicity, deltasync

lunchbox = LunchBox.from_folder("n_700_conn_scan")
logger=logging.getLogger("APPLICATION")

def sync(block):
    return deltasync(block, bootstrap_trials=20)

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


fig, axes = plt.subplots(2,1, sharex=True)

for extr, ax in zip([np.max, np.min], axes):
    val = extr(data[data>0.0])
    for sys_id in lunchbox.systems.keys():
        if lunchbox.extractions[sys_id]['sync']['exc'] == val:
            sizes, deltas, intercept, coef = deltasync(lunchbox.systems[sys_id].pops['exc'], return_all=True, subsamp_sizes=[10,20,30,40,50,60,70,80,90, 100, 150], bootstrap_trials=20)
            plt.plot(1.0/sizes, deltas)
            xx = np.linspace(0,0.1, 10)
            plt.plot(xx, intercept + coef*xx, label=f"sync={val}")
            break
    plt.legend()
plt.show()