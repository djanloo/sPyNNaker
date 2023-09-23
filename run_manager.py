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
import dill as pickle
import time
import multiprocessing as mp
from time import perf_counter


import numpy as np
import pandas as pd
from pyNN.random import RandomDistribution
from spinn_front_end_common.utilities.exceptions import ConfigurationException

from local_utils import get_sim, num, set_logger_pid

import logging
from local_utils import set_loggers;
set_loggers()

LUNCHBOX_PARAMS = ['duration', 'timestep', 'time_scale_factor', 'min_delay', 'neurons_per_core']

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
        self._was_converted = False
        logger.info(f"Successfully created {self} with params\n{self.params_dict}")

    @property
    def id(self):
        if self._id is None:
            self._id =f"{os.getpid()}-{id(self)}"
        return self._id

    @id.setter
    def id(self, value):
        logger.debug(f"Id of {self} was forcefully changed to {value}")
        self._id = value

    def pupate(self):
        bad_populations = []
        for popname in self.pops:
            # Population get substituted with their neo blocks
            try:
                self.pops[popname] = self.pops[popname].get_data()
            except ConfigurationException as e:
                logger.warning(f"An error lead to the deletion of population {popname} af system {self}\nError was: {e}")
                bad_populations += [popname]

        for popname in bad_populations:
            del self.pops[popname]

        self._was_converted = True
    
    def extract(self, function):
        if not self._was_converted:
            raise RuntimeError("System must be converted to neo block first")
        
        extraction = dict()
        for pop in self.pops.keys():
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
        try:
            sys.id = num(os.path.basename(folder))
        except ValueError as e:
            msg = f"Folder {folder} has not an appropriate name, raised: {e}"
            logger.warning(msg)
            raise ValueError(msg)
        
        with open(f"{folder}/conf.cfg", "rb") as confile:
            sys.params_dict = pickle.load(confile)
        
        pop_names = [f.replace(".pkl", "") for f in os.listdir(folder) if f.endswith(".pkl")]
        
        for pop in pop_names:
            with open(f"{folder}/{pop}.pkl", "rb") as popfile:
                sys.pops[pop] = pickle.load(popfile)
                logger.info(f"{sys}: added population <{pop}>")

        sys._was_converted = True
        return sys

class LunchBox:
    """A container for systems that can be run together, 
    i.e. ones sharing duration, timescale and timestep.
    """

    def __init__(self, folder="RMv2", add_old=True,  **box_params):
        
        self.box_params = box_params
        logger.info(f"Initialized run box with params: {self.box_params}")
        sim = get_sim()
        # Sets the simulator
        if sim is not None:
            self.sim = sim
            self.sim_params = {par:box_params[par] for par in ['timestep', 'time_scale_factor', 'min_delay']}
            self.sim.setup(**self.sim_params)
            self.neurons_per_core = box_params['neurons_per_core']
            if sim.__name__ == 'pyNN.spiNNaker':
                logger.info(f"setting {self.neurons_per_core} neurons per core since we are on a spiNNaker machine")
                sim.set_number_of_neurons_per_core(sim.IF_cond_exp, self.neurons_per_core)
        
        try:
            self.duration = box_params['duration']
        except KeyError as e:
            logger.warning("Duration of the lunchbox was not set")
            self.duration = None

        # Dictionary of all the systems of the lunchbox
        # Indexed by id
        self.systems = dict()

        # For function evaluation
        self._extraction_functions = []

        self.folder = folder
        self.add_old = add_old
        try:
            os.mkdir(self.folder)
        except FileExistsError:
            pass

    def add_system(self, system):
        self.systems[system.id] = system
    
    def add_extraction(self, function):
        """Computes a function of each system
        
        To do so, function must take a population as an argument.
        """
        self._extraction_functions.append(function)
    def _convert_to_neo_block(self):
        for system_id in self.systems.keys():
            self.systems[system_id]._convert_to_neo_block()

    def _extract(self):

        logger.info(f"Starting functions extraction ")
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
    
    def get_extraction_triplets(self, param1=None, param2=None, extraction=None):

        extractions_for_each_population = dict()

        # Let me assume that each system has the same populations
        try:
            pops = self.extractions[list(self.systems.keys())[0]][extraction].keys()
        except KeyError as e:
            logger.warning(f"An error triggered the recomputation of the extactions: {e}")
            self._extract()
            pops = self.extractions[list(self.systems.keys())[0]][extraction].keys()

        logger.info(f"Getting extraction triplets ({param1}, {param2}, {extraction}) for populations {list(pops)}")

        for pop in pops:
            extraction_triplet = dict()
            
            extraction_triplet[param1, param2] = []
            extraction_triplet[extraction] = []
            
            for system_id in self.systems.keys():

                param1_value = self.systems[system_id].params_dict[param1]
                param2_value = self.systems[system_id].params_dict[param2]
                extraction_triplet[param1, param2].append([param1_value, param2_value])

                extraction_triplet[extraction].append(self.extractions[system_id][extraction][pop])

            extraction_triplet[param1, param2] = np.array(extraction_triplet[param1, param2])
            extraction_triplet[extraction] = np.array(extraction_triplet[extraction])

            extractions_for_each_population[pop] = extraction_triplet
        return extractions_for_each_population

    def run(self):
        total_neurons = [self.systems[system_id].params_dict['n_neurons'] for system_id in self.systems.keys()]
        total_neurons = np.sum(total_neurons)


        logger.info(f"Running lunchbox composed of {len(self.systems)} systems ({total_neurons} total neurons) for {self.duration} timesteps")
        logger.info(f"Extractions functions are: {[f.__name__ for f in self._extraction_functions]}")
        start = time.perf_counter()
        self.sim.run(self.duration)
        self._run_time = time.perf_counter() - start
        logger.info(f"Simulation took {self._run_time:.1f} seconds")

    def extract_and_save(self, save_pops=False, save_extraction_functions=False):

        # Pupates each system fro population to neo_core block
        for sys in self.systems.values():
            sys.pupate()

        if save_pops:
            self._save_systems()

        self._extract()

        if save_extraction_functions:
            self._save_extraction_functions()

        self._save_results()
    
    def _save_systems(self):

        if self.add_old:
            if os.path.exists(f"{self.folder}/systems.pkl"):
                logger.warning(f"Adding already existing systems...")
                with open(f"{self.folder}/systems.pkl", "rb") as systems_file:
                    existing_systems = pickle.load(systems_file)
                    self.systems.update(existing_systems)
                logger.warning(f"Added {len(existing_systems)} systems")
        else:
            logger.warning(f"Old systems in folder {self.folder} have been overwritten since option add_old is False")
        logger.info("Saving systems...")
        with open(f"{self.folder}/systems.pkl", "wb") as systems_file:
            pickle.dump(self.systems, systems_file)

    def _save_results(self):
        
        with open(f"{self.folder}/extractions.pkl", "wb") as extr_file:
            logger.info(f"saving LunchBox extractions in {self.folder}/extractions.pkl")
            pickle.dump(self.extractions, extr_file)

        with open(f"{self.folder}/lunchbox_conf.pkl", "wb") as boxpar_file:
            logger.info(f"saving LunchBox configuration {self.folder}/lunchbox_conf.pkl")
            pickle.dump(self.box_params, boxpar_file)
    
    def _save_extraction_functions(self):
        with open(f"{self.folder}/extractions_functions.pkl", "wb") as extrf_file:
            logger.info(f"saving LunchBox extractions functions in {self.folder}/extractions_functions.pkl")
            pickle.dump(self._extraction_functions, extrf_file)

    def get_systems_in_region(self, extrema_dict):
        systems_in_region = []
        for sys_id in self.systems.keys():
            is_in_region = True
            params =self.systems[sys_id].params_dict
            for par in extrema_dict.keys():
                try:
                    if params[par] < extrema_dict[par][0] or params[par] > extrema_dict[par][1]:
                        is_in_region = False
                        logger.debug(f"system excluded from region since {par}={params[par]} is not in {extrema_dict[par]}")
                except KeyError as e:
                    logger.warning(f"Parameter {par} was not found in {self.systems[sys_id]}\nRaised: {e}")
                    is_in_region = False
            if is_in_region:
                systems_in_region.append(self.systems[sys_id])
        logger.debug(f"For region {extrema_dict} returning systems having params:")
        for sys in systems_in_region:
            logger.debug(sys.params_dict)
        return systems_in_region


    @classmethod
    def from_folder(cls, folder):
        lunchbox = cls(None, folder=folder)

        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder {folder} does not exist")
        
        with open(f"{folder}/systems.pkl", "rb") as syst_file:
            logger.info("LunchBox: loading systems")
            lunchbox.systems = pickle.load(syst_file)

        with open(f"{folder}/extractions.pkl", "rb") as extr_file:
            logger.info("LunchBox: loading extractions")
            lunchbox.extractions = pickle.load(extr_file)

        with open(f"{folder}/extractions_functions.pkl", "rb") as extrf_file:
            logger.info("LunchBox: loading extraction functions")
            lunchbox._extraction_functions = pickle.load(extrf_file)
        
        return lunchbox


class PanHandler:

    def __init__(self, build_function, folder="pan_handler"):

        self.build_function = build_function  
        self.system_dicts = []
        self.lunchboxes_dicts = []
        self._extraction_functions = []
        self.folder= folder

        try:
            os.mkdir(folder)
        except FileExistsError:
            pass

        self._clean_folder()

    def add_system_dict(self, system_dict):
        self.system_dicts.append(system_dict)

    def add_lunchbox_dict(self, lunchbox_dict):
        self.lunchboxes_dicts.append(lunchbox_dict)

    def add_extraction(self, func):
        self._extraction_functions.append(func)

    def run(self):
        self._run_time = perf_counter()

        logger.info("Starting lunchboxes having:")
        for lbd in self.lunchboxes_dicts:
            logger.info(lbd)

        logger.info("Each lunchbox has systems having params:")
        for sd in self.system_dicts:
            logger.info(sd)


        processes = []
        for lbd in self.lunchboxes_dicts:
            p = mp.Process(target=self.__class__._create_lunchbox_run_and_save, 
                                  args=(self.build_function, 
                                        lbd, 
                                        self.system_dicts,
                                        self._extraction_functions,
                                        self.folder))
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()

        self._extract()
        self._run_time = perf_counter() - self._run_time

        logger.info(f"Whole PanHandler took {self._run_time:.1f} seconds")

    @classmethod
    def _create_lunchbox_run_and_save(cls, 
                                        build_func, 
                                        lunchbox_dict, 
                                        system_dicts,
                                        extractions, 
                                        folder):
        
        set_logger_pid(logger)

        lunchbox_dict['folder'] = os.path.join(folder, str(os.getpid()))

        lb = LunchBox(**lunchbox_dict)
        for sys_dict in system_dicts:
            lb.add_system(System(build_func, sys_dict))
        
        for extr in extractions:
            lb.add_extraction(extr)

        lb.run()
        lb.extract_and_save()

    def _extract(self):
        self.extractions = dict()
        logger.info(f"Starting gathering extractions for PanHandler")
        for subf in os.listdir(self.folder):
            logger.info(f"Gathering {self.folder}/{subf}")
            if os.path.isdir(os.path.join(self.folder, subf)):
                self.extractions[subf] = dict()
                with open(f"{self.folder}/{subf}/extractions.pkl", "rb") as f:
                    self.extractions[subf]['extractions'] = pickle.load(f)
                
                with open(f"{self.folder}/{subf}/lunchbox_conf.pkl", "rb") as f:
                    self.extractions[subf]['lunchbox_conf'] = pickle.load(f)

    def _clean_folder(self):
        logger.info(f"CLeaning folder {self.folder}")
        for subfolder in os.listdir(self.folder):
            if not os.path.isdir(os.path.join(self.folder, subfolder)):
                logger.debug(f"{subfolder} is not a subfolder. skipping.")
                continue

            logger.info(f"Deleting subfolder <{subfolder}>")

            for file in os.listdir(os.path.join(self.folder, subfolder)):
                file_path = os.path.join(self.folder, subfolder, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        os.rmdir(file_path)
                except Exception as e:
                    logger.warning(f"Exception was raised when deleting {file_path}: {e}")
            os.rmdir(os.path.join(self.folder, subfolder))


def data_flattener(dizionario, prefisso=None):
    lista_di_liste = []
    for chiave, valore in dizionario.items():
        chiave_completa = prefisso + (chiave,) if prefisso else (chiave,)
        if isinstance(valore, dict):
            lista_di_liste.extend(data_flattener(valore, chiave_completa))
        else:
            lista_di_liste.append(list(chiave_completa) + [valore])
    return lista_di_liste