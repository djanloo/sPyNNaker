"""A systems of containers for running parameter gridsearch in parallel on SpiNNaker.



    And blood-black nothingness began to spin

    A system of cells interlinked within

    Cells interlinked within cells interlinked

    Within one stem. And dreadfully distinct

    Against the dark, a tall white fountain played.

    
AUTHOR: djanloo
DATE:   12/09/23
"""
import os
import pickle
import numpy as np

from pyNN.random import RandomDistribution

from local_utils import get_sim, num
sim = get_sim()

import logging
from local_utils import set_loggers; set_loggers()
logger = logging.getLogger("RUNMANAGER")


class System:
    """A system is a collection of populations that can interact only among themselves"""
    def __init__(self, build_method, dict_of_params):
        """
        build_method:           a function that returns the <dict> of populations
        dict_of_params:         a dictionary {population: parameters}

        """
        self.build_method = build_method
        self.params_dict = dict_of_params
        self.pops = build_method(dict_of_params)
    
        logger.info(f"Successfully created {self} with params\n{self.params_dict}")

    @property
    def id(self):
        tuple_of_params = tuple(self.params_dict.values())
        # tuple_of_params = tuple([(name, val) for name, val in self.params_dict.items()])
        return hash(tuple_of_params)
    
    def save(self, where):
        try:
            os.mkdir(where)
        except FileExistsError:
            pass

        try:
            os.mkdir(f"{where}/{self.id}")
        except FileExistsError:
            pass
        
        logger.info(f"Saving for {self}")

        for pop in self.pops.keys():
            self.pops[pop].write_data(f"{where}/{self.id}/{pop}.pkl")

        # Saves the configuration of the run
        with open(f"{where}/{self.id}/conf.cfg", 'wb') as file:
            logger.info(f"Saving config file for system {self} population  in {where}/{self.id}/conf.cfg")
            pickle.dump(self.params_dict, file)

    def extract(self, function):
        extraction = dict()
        for pop in self.pops.keys():
            # test = self.pops[pop].get_v().segments[0].analogsignals
            # logger.info(f"test is {test}")
            # logger.info(f"dir(test) is {dir(test)}")
            try:
                extraction[pop] = function(self.pops[pop])
            except Exception as e:
                logger.error(f"Function evaluation on population <{pop}> raised: {e}")
                logger.warning(f"Skipping evaluation of <{function.__name__}> on <{pop}>")

        return extraction


    def __repr__(self):
        return f"< System {id(self)}>"
    
    def __str__(self):
        return repr(self)

class RunBox:
    """A container for systems that can be run together, 
    i.e. ones sharing duration, timescale and timestep.
    """

    def __init__(self, simulator, box_params, folder="RMv2"):

        self.box_params = box_params
        logger.info(f"Initialized run box with params: {self.box_params}")

        # Sets the simulator
        self.sim = simulator
        self.sim_params = {par:box_params[par] for par in ['timestep', 'time_scale_factor', 'min_delay']}
        self.sim.setup(**self.sim_params)
        
        if sim.__name__ == 'pyNN.spiNNaker':
            logger.info("setting 50 neurons per core since we are on a spiNNaker machine")
            sim.set_number_of_neurons_per_core(sim.IF_cond_exp, 50)

        set_loggers()

        self.duration = box_params['duration']

        self.systems = []

        # For function evaluation
        self._extraction_functions = []

        self.folder = folder

    def add_system(self, system):
        self.systems.append(system)
    
    def add_extraction(self, function):
        """Computes a function of each system
        
        To do so, function must take a population as an argument.
        """
        self._extraction_functions.append(function)
    
    def extract(self):
        system_extraction = dict()
        for system in self.systems:
            for function in self._extraction_functions:
                logger.debug(f"Extracting <{function.__name__}> from {system}")
                system_extraction[function.__name__, str(system)] = system.extract(function)
                logger.debug(f"Got dictionary with keys {system_extraction[function.__name__, str(system)].keys()}")
        return system_extraction
    
    def run(self):
        logger.info("Running runbox...")
        self.sim.run(self.duration)
        logger.info("Simulation Done")

    def save(self):
        for system in self.systems:
            system.save(self.folder)

class RunGatherer:

    def __init__(self, folder):
        self.folder =folder
