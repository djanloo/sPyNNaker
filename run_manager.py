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
from time import perf_counter, sleep
import importlib
import json


import numpy as np
import pandas as pd
import pyNN.spiNNaker
from spinn_front_end_common.utilities.exceptions import ConfigurationException
from spinnman.exceptions import SpiNNManCoresNotInStateException

from local_utils import get_sim, num, set_logger_pid

import logging
from local_utils import set_loggers;
set_loggers()

MAX_RUN_ATTEMPTS = 10
WAIT_TIME_S = 350 # seconds
LUNCHBOX_PARAMS = ['duration', 'timestep', 'time_scale_factor', 'min_delay', 'neurons_per_core']

logger = logging.getLogger("RUN_MANAGER")

database = pd.DataFrame()

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
        logger.debug(f"Successfully created {self} with params\n{self.params_dict}")

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
                logger.debug(f"An error lead to the deletion of population {popname} af system {self}\nError was: {e}")
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
                logger.debug(f"Function evaluation on population <{pop}> raised: {e}")
                logger.debug(f"Skipping evaluation of <{function.__name__}> on <{pop}>")
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
            logger.warning(f"An error triggered the recomputation of the extractions: {e}")
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

        self.box_params['run_time'] = self._run_time
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
        get_sim().end()
    
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

        with open(f"{self.folder}/systems_conf.pkl", "wb") as syspar_file:
            sys_params = {sys_id:self.systems[sys_id].params_dict for sys_id in self.systems.keys()}
            logger.info(f"saving Systems configurations {self.folder}/systems_conf.pkl")
            pickle.dump(sys_params, syspar_file)

    
    def _save_extraction_functions(self):
        with open(f"{self.folder}/extractions_functions.pkl", "wb") as extrf_file:
            logger.info(f"saving LunchBox extractions functions in {self.folder}/extractions_functions.pkl")
            pickle.dump(self._extraction_functions, extrf_file)

    # def get_systems_in_region(self, extrema_dict):
    #     systems_in_region = []
    #     for sys_id in self.systems.keys():
    #         is_in_region = True
    #         params =self.systems[sys_id].params_dict
    #         for par in extrema_dict.keys():
    #             try:
    #                 if params[par] < extrema_dict[par][0] or params[par] > extrema_dict[par][1]:
    #                     is_in_region = False
    #                     logger.debug(f"system excluded from region since {par}={params[par]} is not in {extrema_dict[par]}")
    #             except KeyError as e:
    #                 logger.warning(f"Parameter {par} was not found in {self.systems[sys_id]}\nRaised: {e}")
    #                 is_in_region = False
    #         if is_in_region:
    #             systems_in_region.append(self.systems[sys_id])
    #     logger.debug(f"For region {extrema_dict} returning systems having params:")
    #     for sys in systems_in_region:
    #         logger.debug(sys.params_dict)
    #     return systems_in_region


    @classmethod
    def from_folder(cls, folder):
        lunchbox = cls(None, folder=folder)

        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder {folder} does not exist")
        
        with open(f"{folder}/systems.pkl", "rb") as syst_file:
            logger.info("LunchBox: loading systems")
            lunchbox.systems = pickle.load(syst_file)

        with open(f"{folder}/extractions_functions.pkl", "rb") as extrf_file:
            logger.info("LunchBox: loading extraction functions")
            lunchbox._extraction_functions = pickle.load(extrf_file)

        try:
            with open(f"{folder}/extractions.pkl", "rb") as extr_file:
                logger.info("LunchBox: loading extractions")
                lunchbox.extractions = pickle.load(extr_file)
        except FileExistsError:
            logger.warning("During creation of lunchbox from folder {folder}, file 'extractions.pkl' was not found.")
        
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
        self._check_lunchboxes()
        self._run_time = perf_counter()

        logger.info(f"Starting {len(self.lunchboxes_dicts)} lunchboxes having:")
        for lbd in self.lunchboxes_dicts:
            logger.info(lbd)

        logger.info("Each lunchbox has systems having params:")
        for sd in self.system_dicts:
            logger.info(sd)

        self._init_subruns()

        counter = 0
        all_done = False
        while counter <= MAX_RUN_ATTEMPTS and not all_done:
            counter +=1
            logger.info(f"Run attempt: {counter}")
            all_done = self._run_empty()

        # Endind instructions
        self._extract()
        self._run_time = perf_counter() - self._run_time

        logger.info(f"Whole PanHandler took {self._run_time:.1f} seconds")

    def _init_subruns(self):
        logger.info("Initializing lunchboxes folders")
        for i, lbd in enumerate(self.lunchboxes_dicts):
            
            lbd['folder'] = os.path.join(self.folder, str(i))
            os.mkdir(lbd['folder'])

            with open(os.path.join(lbd['folder'], 'retry_file.pkl'), "wb") as retry_file:
                    retry_dict = dict(build_func=self.build_function, 
                                       lunchbox_dict=lbd, 
                                       system_dicts=self.system_dicts, 
                                       extractions=self._extraction_functions,
                                       folder=self.folder,
                                       subfolder=lbd['folder'])
                    pickle.dump(retry_dict, retry_file)

    def _run_empty(self):
        logger.info("Running empty folders")
        empty_run_folders = self._check_empty_runs()

        if len(empty_run_folders) == 0:
            logger.info("No empty folders found to run.")
            return True
        
        processes = []
        for empty_run_folder in empty_run_folders:
            with open(os.path.join(self.folder, empty_run_folder, 'retry_file.pkl'), "rb") as retry_file:
                creation_args = pickle.load(retry_file)

            creation_args['subfolder'] = empty_run_folder

            p = mp.Process(target=self.__class__._create_lunchbox_run_and_save, 
                                  kwargs=creation_args)
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()
        
        return False
        
    def _check_empty_runs(self):
        subfolders = [subf for subf in os.listdir(self.folder) if os.path.isdir(os.path.join(self.folder, subf ))]

        subfolder_is_ok = dict()
        for subfolder in subfolders:
            subfolder_is_ok[subfolder] = os.path.exists(os.path.join(self.folder, subfolder, "extractions.pkl"))

        if not np.all(np.array(list(subfolder_is_ok.values()))):
            logger.warning(f"Bad-run check found valid folders:\n{subfolder_is_ok}")

        return [key for key in subfolder_is_ok.keys() if not subfolder_is_ok[key]]

    def _check_lunchboxes(self):
        logger.info(f"Checking {len(self.lunchboxes_dicts)} lunchboxes...")

        is_duplicate = np.zeros(len(self.lunchboxes_dicts), int)

        for i in range(len(self.lunchboxes_dicts)):
            for j in range(i+1, len(self.lunchboxes_dicts)):
                if self.lunchboxes_dicts[i] == self.lunchboxes_dicts[j]:
                    is_duplicate[j] = 1

        if np.sum(is_duplicate) > 0:
            logger.warning(f"Found {np.sum(is_duplicate)} duplicate lunchboxes.\nConsider adding duplicate systems to optimize computing.")

    @classmethod
    def _create_lunchbox_run_and_save(cls, 
                                        build_func=None, 
                                        lunchbox_dict=None, 
                                        system_dicts=None,
                                        extractions=None, 
                                        folder=None, subfolder=None):
        
        set_logger_pid(logger)

        localized_lunchbox_dict = lunchbox_dict.copy()
        localized_lunchbox_dict['folder'] = os.path.join(folder, str(os.getpid())) if subfolder is None else os.path.join(folder, subfolder)

        try:
            # Try a reload of spynnaker
            # because spalloc has timeouts
            importlib.reload(pyNN.spiNNaker)

            lb = LunchBox(**localized_lunchbox_dict)

            for sys_dict in system_dicts:
                lb.add_system(System(build_func, sys_dict))
            
            for extr in extractions:
                lb.add_extraction(extr)

            lb.run()
            lb.extract_and_save()

        except SpiNNManCoresNotInStateException:
            logger.error(f"Not enough free cores. Execution queued.")
            with open(os.path.join(localized_lunchbox_dict['folder'], 'retry_file.pkl'), "wb") as retry_file:
                retry_dict = dict(build_func=build_func, 
                                    lunchbox_dict=lunchbox_dict, 
                                    system_dicts=system_dicts, 
                                    extractions=extractions,
                                    folder=folder)
                pickle.dump(retry_dict, retry_file)
            raise SpiNNManCoresNotInStateException("Not enough free cores.")

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
        logger.info(f"Cleaning folder {self.folder}")
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


class DataGatherer:

    def __init__(self, folder):
        self.folder = folder

        self.subfolders = [subf for subf in os.listdir(self.folder) if os.path.isdir(os.path.join(self.folder, subf ))]
        logger.info(f"Gatherer has found subfolders:\n {self.subfolders}")

        self.valid_subfolders = []
        for sub in self.subfolders:
            valid = True
            sub_path = os.path.join(self.folder, sub)
            for file in ["lunchbox_conf.pkl","systems_conf.pkl", "extractions.pkl"]:
                if not os.path.exists(os.path.join(sub_path, file)):
                    valid = False
            if valid:
                self.valid_subfolders.append(sub)
        
        if len(set(self.subfolders) - set(self.valid_subfolders)) > 0:
            logger.warning(f"Found {len(set(self.subfolders) - set(self.valid_subfolders))} not valid subfolders. Skipping.")

        self.database = pd.DataFrame()

    def gather(self):

        for sub in self.valid_subfolders:

            sub_path = os.path.join(self.folder, sub)

            ### Infos abot the lunchbox
            with open(os.path.join(sub_path, "lunchbox_conf.pkl"), "rb") as conf_file:
                lunchbox_params = pickle.load(conf_file)
            lunchbox_params['lunchbox_id'] = sub
            logger.debug(f"Found lunchbox {sub} with params: \n{lunchbox_params}")

            ## Infos about systems
            with open(os.path.join(sub_path, "systems_conf.pkl"), "rb") as sysfile:
                sys_params = pickle.load(sysfile)

            ## Infos about extractions
            with open(os.path.join(sub_path, "extractions.pkl"), "rb") as extr:
                extr_dict = pickle.load(extr)
            
            logger.debug(f"For lunchbox {sub} found extractions: \n{extr_dict}")

            for sys_id in extr_dict.keys():
                for func in extr_dict[sys_id].keys():
                    for pop in extr_dict[sys_id][func].keys():
                        # Check for multiple results
                        if isinstance(extr_dict[sys_id][func][pop], dict):
                            for inside_func in extr_dict[sys_id][func][pop].keys():
                                row = dict()
                                row['sys_id'] = sys_id
                                row['func'] = inside_func
                                row['pop'] = pop
                                row['extraction'] =  extr_dict[sys_id][func][pop][inside_func]


                                row.update(lunchbox_params)
                                row.update(sys_params[sys_id])

                                row = pd.DataFrame(row, index=[0])
                                self.database = pd.concat([self.database, row], ignore_index=True)
                        else:
                            row = dict()
                            row['sys_id'] = sys_id
                            row['func'] = func
                            row['pop'] = pop
                            row['extraction'] =  extr_dict[sys_id][func][pop]


                            row.update(lunchbox_params)
                            row.update(sys_params[sys_id])

                            row = pd.DataFrame(row, index=[0])
                            self.database = pd.concat([self.database, row], ignore_index=True)

        logger.info(f"Built database: \n{self.database}")
        self.database.to_csv("A.csv")
        return self.database
    
    def adjust_db(self):
        raise NotImplementedError("TODO")
        self.adjusted_df = pd.DataFrame()
        self.database['id'] = self.database.sys_id.apply(str) + "_" + self.database['pop']
        self.database = self.database.set_index("id")

        df['id'] = np.unique(self.database.index.values)
        df['pop_type'] = df['id'].apply( lambda x: x[-3:])

        df = df.set_index('id')

        funcs = np.unique(self.database.func)

        time_series_funcs = []
        for func in funcs:
            df[func] = self.database[ self.database.func == func].extraction
            try:
                df[func] = df[func].astype(float)
            except Exception as e:
                time_series_funcs.append(func)

        for column in ['timestep', 'exc_conn_p']:
            df[column] = self.database[self.database.func == funcs[0]][column]
            
            
        df['is_asleep'] = (df.avg_activity < 2.0).astype(int)

        df.rate_of_active_avg = df.rate_of_active_avg.fillna(0)
        df.rate_of_active_std = df.rate_of_active_std.fillna(0)