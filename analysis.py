import argparse
import os
import pickle
import numpy as np
from quantities import mV, nA

import matplotlib.pyplot as plt

import logging
from local_utils import  avg_activity_by_spiketrains, set_loggers
from plotting import annotate_dict
from plotting import PlotGroup, DensityPlot, SpikePlot, QuantilePlot
import seaborn as sns

logger = logging.getLogger("ANALYSIS")
set_loggers()

possible_plots = ["quantiles", "density", "spikes"]
quantities = ["v", "gsyn_exc", "gsyn_inh", "I"]

def read_neo_file(folder_name, file):
    """
    file name must be given without the extension.
    """
    data = dict()

    with (open(f"{folder_name}/{file}.pkl", "rb")) as openfile:
        neo_blocks = []
        while True:
            try:
                neo_blocks.append(pickle.load(openfile))
            except EOFError:
                break

    if len(neo_blocks) > 1:
        logger.critical(f"more than one neo blocks were found in file {file}")
        raise NotImplementedError("Only one neo block can be contained in each serialized object.")

    # Analog signals
    for analogsignal in neo_blocks[0].segments[0].analogsignals:
        data[file, analogsignal.name] = analogsignal
        logger.debug(f"In file: {file:20} found signal: {analogsignal.name:20}")

    # Spike trains
    data[file, "spikes"] = neo_blocks[0].segments[0].spiketrains
    logger.debug(f"In file: {file:20} found signal: spikes")

    # Adds currents
    try:
        data[file, "I"] = (data[file,"v"] - 0*mV)*data[file, 'gsyn_exc'] \
                            + (data[file, "v"] - (-80*mV))*data[file, 'gsyn_inh']
        data[file, 'I'] = data[file, 'I'].rescale(nA)
        logger.debug(f"Current has units {data[file, 'I'].units}")
    except KeyError:
        logger.warning(f"Could not determine current for population {file}.")

    return data


def system_analysis(args):
    logger.info(f"Starting system analysis with args {args}")

    system_name = args['folder']

    plot_groups = []

    for population in args['pops']:

        plot_groups.append(PlotGroup(population))

        # Reads the file
        try:
            data = read_neo_file(system_name, population)
        except FileNotFoundError:
            logger.warning(f"File {population}.pkl does not exist. Skipping.")
            # Skips the file if the file is not found
            continue
        
        ######################### SIMULATION PARAMETERS #########################
        if args['conf'] is None:
            config_file = "conf"
            logger.warning(f"Configuration file was automatically set to {config_file}.cfg")

        try:

            with open(f"{system_name}/{config_file}.cfg", "rb") as f:
                conf_dict = pickle.load(f)
                logger.info(f"Configuration of the run for {population} is {conf_dict}")

        except FileNotFoundError:
            logger.warning(f"Configuration file not found in {system_name}/{config_file}.cfg. Setting empty run informations.")
            conf_dict = dict(config_file="?")
        
        ################################## PLOTTING ##################################
        
        # Spikes & activity
        if "spikes" in args['plot']:

            sp = SpikePlot(data[population, 'spikes'], args['bins'])

            avg_act = avg_activity_by_spiketrains(conf_dict['n_neurons'], data[population, 'spikes'])

            # Infos
            annotate_dict(conf_dict, sp.axes['infos'])
            sp.fig.suptitle(population)
            plot_groups[-1].add_fig(sp)


        # V-density
        if "density" in args['plot']:

            try:
                _ = data[population, args['quantity']]
            except KeyError:
                logger.warning(f"Plot for population {population} was skipped: population has no {args['quantity']}")
            else:
                dp = DensityPlot(data[population, args['quantity']], 
                                args['bins'])
                # Infos
                # dp.fig.suptitle(fr"$\rho(V, t)$ for {population}")
                annotate_dict(conf_dict, dp.axes['infos'])
                dp.fig.suptitle(population)
                plot_groups[-1].add_fig(dp)
            

        # Quantiles
        if "quantiles" in args['plot']:
            try:
                last_plot = plot_groups[-1].plots[-1]
                if isinstance(last_plot, DensityPlot):
                    fig = last_plot.fig
                    axes = last_plot.axes
                else:
                    fig, axes = None, None
            except IndexError:
                fig, axes = None, None
            try:
                _ = data[population, args['quantity']]
            except KeyError:
                logger.warning(f"Plot for population {population} was skipped: population has no {args['quantity']}")
            else:
                qp = QuantilePlot(data[population, args['quantity']], fig=fig, axes=axes)

    for pg in plot_groups:
        pg.save(f"{system_name}/outputs")


def runbox_analysis(args):
    subfolders = [dir_ for dir_ in os.listdir(os.path.abspath(os.path.join("./", args['folder'])))]

    logger.info(f"Found {len(subfolders)} subfolders to analyse in {args['folder']}: {subfolders}")
    for subfolder in subfolders:
        files = [f.replace(".pkl", "") for f in os.listdir(f"{args['folder']}/{subfolder}") if f.endswith(".pkl")]
        system_args = args.copy()
        system_args['folder'] = f"{args['folder']}/{subfolder}"
        system_args['pops'] = files
        system_analysis(system_args)

    # On the remote server save instead of showing
    if os.environ.get("USER") != "bbpnrsoa":
        plt.show()

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Analisys of simulation files')


    parser.add_argument('--pops', 
                        nargs='+', action='append',
                        default=None,
                        help='the populations under analysis')

    parser.add_argument('--folder', 
                        type=str, 
                        default="VA_data",
                        help='the folder of the simulation data')

    parser.add_argument('--plot', 
                        type=str, 
                        choices=possible_plots, 
                        nargs='+', 
                        default="spikes",
                        help=f"plot to be displayed, choices are: {', '.join(possible_plots)}")

    parser.add_argument('--bins', 
                        type=int,
                        default=30,
                        help="the number of bins for activity plot"
                        )
    parser.add_argument('--quantity',
                    type=str,
                    default='v',
                    choices=quantities,
                    help=f"the quantity to plot, chosed among: {', '.join(quantities)}"
                    )

    parser.add_argument('--v', 
                        type=int, 
                        default=1,
                        help="verbosity level")
                    
    parser.add_argument('--list_files', 
                        action="store_true",
                        help="list available files")
    parser.add_argument('--all',
                        default=True,
                        action='store_true',
                        help='plots data for each file in the folder')

    parser.add_argument('--conf', type=str, default=None, help="the configuration file of the run")

    args = parser.parse_args()

    # Gets the logger for here
    set_loggers(args.v)
    logger = logging.getLogger("ANALYSIS")

    # Strips the backslash from the folder name
    folder_name = args.folder.replace('/', '')

    if args.list_files or args.pops is None:
        data=dict()
        files = [f.replace(".pkl", "") for f in os.listdir(folder_name) if f.endswith(".pkl")]
        for file in files:
            read_neo_file(file)


    runbox_analysis(vars(args))