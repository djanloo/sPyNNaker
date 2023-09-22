import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from sklearn.manifold import TSNE
from umap import UMAP

from run_manager import LunchBox, System
from vogels_abbott import build_system

from local_utils import set_loggers; set_loggers()
logger = logging.getLogger("APPLICATION")

from local_utils import get_sim; 
sim = get_sim()

from local_utils import avg_activity, spiketrains_to_couples
DURATION = 3000
SEED = 31121997

params = dict(n_neurons=1501, 
            exc_conn_p=0.03, 
            inh_conn_p=0.02,
            synaptic_delay=2
            )

lunchbox = LunchBox(sim, timestep=1, 
                    time_scale_factor=50, # Defines the lunchbox where the systems will be runned on
                    duration=DURATION, 
                    min_delay=2,
                    rng_seeds=[SEED],
                    neurons_per_core=250,
                    folder=f"n_{params['n_neurons']}_conn_scan",
                    add_old = False
                )

lunchbox.add_system(System(build_system, params))

# Start the simulation & save
lunchbox.run()
lunchbox.extract_and_save()
##################################

for sys_id in lunchbox.systems.keys():
    pass
v = lunchbox.systems[sys_id].pops['exc'].segments[0].filter(name="v")[0].magnitude
avg_act = avg_activity(lunchbox.systems[sys_id].pops['exc'])
spikes = lunchbox.systems[sys_id].pops['exc'].segments[0].spiketrains

logger.info(f"Got v of shape {v.shape}")
logger.info(f"Activity is  {avg_act}")

logger.info("Starting embedding")
reducer = UMAP(n_neighbors=60, 
               min_dist=0.01, 
            #    random_state=SEED,
               )
embedding = reducer.fit_transform(v)

logger.info(f"Got embedding with shape {embedding.shape}")

viridis_colors = mpl.colormaps['viridis'].resampled(DURATION - 50).colors
all_colors = np.concatenate((50*[[0.8, 0.0, 0.0, 1.0]], viridis_colors))
logger.info(f"Creating colormap:\n{all_colors}")
custom_cmap = ListedColormap(all_colors)

mappable = plt.scatter(*(embedding.T), c=np.linspace(0, DURATION, v.shape[0]), s=2, alpha=0.8, cmap=custom_cmap)
plt.colorbar(mappable, label="Time [ms]")
plt.axis("off")

plt.figure(2)
plt.scatter(*(np.flip(spiketrains_to_couples(spikes).T)), c=np.flip(spiketrains_to_couples(spikes).T)[0], vmin=0, vmax=DURATION, s=1, alpha=0.5, cmap=custom_cmap)
plt.show()
