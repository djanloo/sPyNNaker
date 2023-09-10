import os

import matplotlib.pyplot as plt
plt.style.use("./style.mplstyle")

from local_utils import spiketrains_to_couples
import numpy as np
import seaborn as sns
import logging


logger = logging.getLogger("PLOTTING")

def annotate_dict(dict_, ax):
    ys = np.linspace(0,1, len(dict_)+2)
    for key, y in zip(dict_.keys(), ys[1:-1]):
        ax.annotate(f"{key}={dict_[key]}", (0, y), ha='center')
    
    ax.set_xlim(-1,1)
    ax.set_ylim(0,1)
    ax.axis('off')

class PlotGroup:

    def __init__(self, name):
        self.name = name
        self.plots = []
    
    def add_fig(self, figure):
        self.plots.append(figure)
    
    def save(self, root_folder, fmt="png"):
        try:
            os.mkdir(f"{root_folder}")
        except FileExistsError:
            pass

        try:
            os.mkdir(f"{root_folder}/{self.name}")
        except FileExistsError:
            pass


        for plot in self.plots:
            logger.info(f"saving plot in {root_folder}/{self.name}/{plot.name}.{fmt}")
            plot.fig.savefig(f"{root_folder}/{self.name}/{plot.name}.{fmt}")


class QuantilePlot:

    name = "quantile_plot"

    def __init__(self, signal, fig=None, axes=None):
        
        if fig is None or axes is None:
            self.fig, self.axes = plt.subplot_mosaic([['analog'],['infos']],
                                                    height_ratios=[1, 0.3],
                                                    figsize=(6,5))
        else:
            self.axes = axes
            self.fig = fig

        qq = np.quantile(np.array(signal), [.25, .5, .75 ], axis=1)
        colors = sns.color_palette("Accent", n_colors=qq.shape[0])

        for q,c,l in zip(qq, colors, range(1,10)):
            self.axes['analog'].plot(q, color=c,label=f"{l*10}-percentile")
        self.axes['analog'].plot(np.mean(signal, axis=1), color="k", label=f"mean", ls=":")
        self.axes['analog'].legend(ncols=3, fontsize=8)

class DensityPlot:

    name = "density_plot"

    def __init__(self, signal, n_bins, fig=None, axes=None):

        if fig is None or axes is None:
            self.fig, self.axes = plt.subplot_mosaic([['analog'],['infos']],
                                                    height_ratios=[1, 0.3],
                                                    figsize=(6,5))
        else:
            self.axes = axes
            self.fig = fig

        logger.debug(f"signal has shape {signal.shape}")

        hist=np.zeros((n_bins, len(signal)))

        v_bins = np.linspace(np.min(signal), np.max(signal), n_bins+1)
        X, Y = np.linspace(0,len(signal), len(signal)), v_bins[:-1]
        X, Y = np.meshgrid(X,Y)

        for time_index in range(len(signal)):
            hist[:, time_index] = np.histogram(signal[time_index], bins=v_bins, density=True)[0]
        hist[hist == 0] = np.min(hist[hist != 0])
        hist = np.log10(hist)
        logger.info(f"in log density (-np.inf)-valued areas have been replaced with value {np.min(hist[np.isfinite(hist)])}")
        
        cbar = self.fig.colorbar(
                                    self.axes['analog'].contourf(X, Y, hist, levels=10)
                                )
        cbar.set_label('log density', rotation=270, size=10)

        # Details
        self.axes['analog'].set_xlabel("t [ms]")
        self.axes['analog'].set_ylabel(f"[{signal.units}]")

        self.axes['analog'].set_ylim(np.min(np.quantile(signal.magnitude, 0.05, axis=1)), 
                                     np.max(np.quantile(signal.magnitude, 0.95, axis=1)))

class SpikePlot:

    name = "spike_plot"

    def __init__(self, spike_train_list, n_bins, fig=None, axes=None):

        if fig is None or axes is None:
            self.fig, self.axes = plt.subplot_mosaic([["spikes", "neuron_activity"], ["time_activity", "infos"]],
                                                            height_ratios=[1,0.4],
                                                            width_ratios=[1, 0.4],
                                                            figsize=(6,5), 
                                                            sharex=False, 
                                                            sharey=False,
                                                            constrained_layout=True)
        else:
            self.fig = fig
            self.axes = axes
        
        spike_array = spiketrains_to_couples(spike_train_list)
        # Spikes
        self.axes['spikes'].scatter(*(np.flip(spike_array.T)), marker=".", color="k", alpha=0.1)
        self.axes['spikes'].set_xlim((spike_train_list.t_start, spike_train_list.t_stop))

        # Activity in time
        act_t = np.histogram(spike_array.T[1], bins=np.linspace(spike_train_list.t_start, spike_train_list.t_stop, n_bins +1), density=False)[0]
        self.axes['time_activity'].step(np.linspace(spike_train_list.t_start, spike_train_list.t_stop, n_bins), act_t)
        self.axes['time_activity'].axhline(np.mean(act_t), ls=":", color='k')
        # Details
        self.axes['time_activity'].set_xlabel("t [ms]")
        self.axes['time_activity'].set_ylabel("PSTH")

        # Activity in neuron

        # Counts how many times each neuron has fired
        fired_neuron_index, n_firings_for_each_neuron = np.unique(spike_array.T[0], return_counts=True)
        logger.debug(f"neurons that fired are {len(fired_neuron_index)} in total")
        logger.debug(f"neurons firing occurrencies are \n{n_firings_for_each_neuron}")
        logger.debug(f"completely inactive neurons are {len(spike_train_list) - len(fired_neuron_index)}")

        # Adds the counts for those that never fired
        n_firings_for_each_neuron = np.concatenate(
                                                (n_firings_for_each_neuron, np.zeros(len(spike_train_list) - len(fired_neuron_index)))  
                                            )

        # Counts how many neurons had the same number of activations
        number_of_activations, number_of_neurons = np.unique(n_firings_for_each_neuron, return_counts=True)
        self.axes['neuron_activity'].barh(number_of_activations, number_of_neurons)
        self.axes['neuron_activity'].set_xscale("log")
        # Details
        self.axes['neuron_activity'].set_xlabel("# of neurons")
        self.axes['neuron_activity'].set_ylabel("# of activations")

        self.axes['infos'].axis("off")
