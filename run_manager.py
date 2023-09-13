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
import time

import numpy as np
from pyNN.random import RandomDistribution

from local_utils import get_sim, num
sim = get_sim()

import logging
from local_utils import set_loggers;
set_loggers()

logger = logging.getLogger("RUN_MANAGER")

class System:
    """A system is a collection of populations that can interact only among themselves"""
    def __init__(self, build_method, dict_of_params):
        """
        build_method:           a function that returns the <dict> of populations
        dict_of_params:         a dictionary {population: parameters}

        """
        self.build_method = build_method
        self.params_dict = dict_of_params
        if build_method is not None:
            self.pops = build_method(dict_of_params)
        else:
            logger.warning("System was initialized without build method")
            self.pops = dict()
        self._id = None

        logger.info(f"Successfully created {self} with params\n{self.params_dict}")

    @property
    def id(self):
        if self._id is None:
            self._id = id(self)
        return self._id

    @id.setter
    def id(self, value):
        logger.warning(f"Id of {self} was forcefully changed to {value}")
        self._id = value

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
        return f"<System {self.id}>"
    
    def __str__(self):
        return repr(self)
    
    @classmethod
    def from_folder(cls, folder):
        sys = cls(None, None)
        sys.id = num(os.path.basename(folder))

        with open(f"{folder}/conf.cfg", "rb") as confile:
            sys.params_dict = pickle.load(confile)
        
        pop_names = [f.replace(".pkl", "") for f in os.listdir(folder) if f.endswith(".pkl")]
        
        for pop in pop_names:
            with open(f"{folder}/{pop}.pkl", "rb") as popfile:
                sys.pops[pop] = pickle.load(popfile)
                logger.info(f"{sys}: added population <{pop}>")
        return sys

class RunBox:
    """A container for systems that can be run together, 
    i.e. ones sharing duration, timescale and timestep.
    """

    def __init__(self, simulator, folder="RMv2",  **box_params):

        self.box_params = box_params
        logger.info(f"Initialized run box with params: {self.box_params}")

        # Sets the simulator
        if simulator is not None:
            self.sim = simulator
            self.sim_params = {par:box_params[par] for par in ['timestep', 'time_scale_factor', 'min_delay']}
            self.sim.setup(**self.sim_params)
            
            if sim.__name__ == 'pyNN.spiNNaker':
                logger.info("setting 50 neurons per core since we are on a spiNNaker machine")
                sim.set_number_of_neurons_per_core(sim.IF_cond_exp, 50)
        
        try:
            self.duration = box_params['duration']
        except KeyError as e:
            logger.warning("DUration of the runbox was not set")
            self.duration = None

        # Dictionary of all the systems of the runbox
        # Indexed by id
        self.systems = dict()

        # For function evaluation
        self._extraction_functions = []

        self.folder = folder

    def add_system(self, system):
        self.systems[system.id] = system
    
    def add_extraction(self, function):
        """Computes a function of each system
        
        To do so, function must take a population as an argument.
        """
        self._extraction_functions.append(function)
    
    def _extract(self):
        logger.info("Staring extraction of functions")
        self.extractions = dict()
        for system_id in self.systems.keys():
            self.extractions[system_id] = dict()
            for function in self._extraction_functions:
                logger.debug(f"Extracting <{function.__name__}> from {self.systems[system_id]}")
                self.extractions[system_id][function.__name__] = self.systems[system_id].extract(function)
                logger.debug(f"Got dictionary with keys {self.extractions[system_id][function.__name__].keys()}")
    
    def get_extraction_couples(self, param=None, extraction=None):

        extractions_for_each_population = dict()

        # Let me assume that each system has the same populations
        for system_id in self.systems.keys():
            pops = self.extractions[system_id][extraction].keys()
            break
        
        logger.info(f"Getting extraction couple ({param}, {extraction}) for populations {list(pops)}")

        for pop in pops:
            extractions_couple = dict()
            
            extractions_couple[param] = []
            extractions_couple[extraction] = []
            
            for system_id in self.systems.keys():
                param_value = self.systems[system_id].params_dict[param]
                extractions_couple[param].append(param_value)
                extractions_couple[extraction].append(self.extractions[system_id][extraction][pop])

            argsort = np.argsort(extractions_couple[param])
            extractions_couple[param] = np.array(extractions_couple[param])[argsort]
            extractions_couple[extraction] = np.array(extractions_couple[extraction])[argsort]

            extractions_for_each_population[pop] = extractions_couple
        return extractions_for_each_population

    def run(self):
        total_neurons = [self.systems[system_id].params_dict['n_neurons'] for system_id in self.systems.keys()]
        total_neurons = np.sum(total_neurons)


        logger.info(f"Running runbox composed of {len(self.systems)} systems ({total_neurons} total neurons) for {self.duration} timesteps")
        start = time.perf_counter()
        self.sim.run(self.duration)
        self._run_time = time.perf_counter() - start
        logger.info(f"Simulation took in {self._run_time} seconds")
        self._extract()

    def save(self):
        for system_id in self.systems.keys():
            self.systems[system_id].save(self.folder)
        
        with open(f"{self.folder}/extractions.pkl", "wb") as extr_file:
            logger.info("saving RunBox extractions")
            pickle.dump(self.extractions, extr_file)

        with open(f"{self.folder}/runbox_conf.pkl", "wb") as boxpar_file:
            logger.info("saving RunBox configuration")
            pickle.dump(self.box_params, boxpar_file)
    
    @classmethod
    def from_folder(cls, folder):
        runbox = cls(None, folder=folder)
        
        with open(f"{folder}/extractions.pkl", "rb") as extr_file:
            logger.info("RunBox: loading extractions")
            runbox.extractions = pickle.load(extr_file)

        subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]

        logger.info(f"Found {len(subfolders)} subfolders to analyse in {folder}: {subfolders}")
        for subfolder in subfolders:
            logger.info(f"RunBox: loading system from folder {subfolder}")
            sys = System.from_folder(f"{subfolder}")
            runbox.add_system(sys)

        return runbox
