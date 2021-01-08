# Copyright (c) 2017-2019 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
from enum import Enum
import os

from pacman.executor.injection_decorator import inject_items
from spinn_utilities.overrides import overrides
from spinn_front_end_common.abstract_models import (
    AbstractGeneratesDataSpecification, AbstractRewritesDataSpecification)
from spinn_front_end_common.utilities.constants import BYTES_PER_WORD
from spinn_front_end_common.utilities import globals_variables
from spynnaker.pyNN.exceptions import SynapticConfigurationException
from spynnaker.pyNN.models.abstract_models import (
    ReceivesSynapticInputsOverSDRAM, SendsSynapticInputsOverSDRAM)
from spynnaker.pyNN.utilities.utility_calls import get_n_bits
from .population_machine_common import CommonRegions, PopulationMachineCommon
from .population_machine_neurons import (
    NeuronRegions, PopulationMachineNeurons, NeuronProvenance)

# Size of SDRAM params = 1 word for address + 1 word for size
# + 1 word for n_neurons + 1 word for n_synapse_types
# + 1 word for number of synapse vertices
# + 1 word for number of neuron bits needed
SDRAM_PARAMS_SIZE = 6 * BYTES_PER_WORD


class PopulationNeuronsMachineVertex(
        PopulationMachineCommon,
        PopulationMachineNeurons,
        AbstractGeneratesDataSpecification,
        AbstractRewritesDataSpecification,
        ReceivesSynapticInputsOverSDRAM):
    """ A machine vertex for the Neurons of PyNN Populations
    """

    __slots__ = [
        "__change_requires_neuron_parameters_reload",
        "__key",
        "__sdram_partition"]

    class REGIONS(Enum):
        """Regions for populations."""
        SYSTEM = 0
        PROVENANCE_DATA = 1
        PROFILING = 2
        RECORDING = 3
        NEURON_PARAMS = 4
        NEURON_RECORDING = 5
        SDRAM_EDGE_PARAMS = 6

    # Regions for this vertex used by common parts
    COMMON_REGIONS = CommonRegions(
        system=REGIONS.SYSTEM.value,
        provenance=REGIONS.PROVENANCE_DATA.value,
        profile=REGIONS.PROFILING.value,
        recording=REGIONS.RECORDING.value)

    # Regions for this vertex used by neuron parts
    NEURON_REGIONS = NeuronRegions(
        neuron_params=REGIONS.NEURON_PARAMS.value,
        neuron_recording=REGIONS.NEURON_RECORDING.value
    )

    _PROFILE_TAG_LABELS = {
        0: "TIMER_NEURONS"}

    def __init__(
            self, resources_required, label, constraints, app_vertex,
            vertex_slice):
        """
        :param ~pacman.model.resources.ResourceContainer resources_required:
            The resources used by the vertex
        :param str label: The label of the vertex
        :param list(~pacman.model.constraints.AbstractConstraint) constraints:
            Constraints for the vertex
        :param AbstractPopulationVertex app_vertex:
            The associated application vertex
        :param ~pacman.model.graphs.common.Slice vertex_slice:
            The slice of the population that this implements
        """
        super(PopulationNeuronsMachineVertex, self).__init__(
            label, constraints, app_vertex, vertex_slice, resources_required,
            self.COMMON_REGIONS,
            NeuronProvenance.N_ITEMS,
            self._PROFILE_TAG_LABELS, self.__get_binary_file_name(app_vertex))
        self.__key = None
        self.__change_requires_neuron_parameters_reload = False
        self.__sdram_partition = None

    @property
    @overrides(PopulationMachineNeurons._key)
    def _key(self):
        return self.__key

    @overrides(PopulationMachineNeurons._set_key)
    def _set_key(self, key):
        self.__key = key

    @property
    @overrides(PopulationMachineNeurons._neuron_regions)
    def _neuron_regions(self):
        return self.NEURON_REGIONS

    def set_sdram_partition(self, sdram_partition):
        """ Set the SDRAM partition.  Must only be called once per instance

        :param ~pacman.model.graphs.machine\
                .SourceSegmentedSDRAMMachinePartition sdram_partition:
            The SDRAM partition to receive synapses from
        """
        if self.__sdram_partition is not None:
            raise SynapticConfigurationException(
                "Trying to set SDRAM partition more than once")
        self.__sdram_partition = sdram_partition

    @staticmethod
    def __get_binary_file_name(app_vertex):
        """ Get the local binary filename for this vertex.  Static because at
            the time this is needed, the local app_vertex is not set.

        :param AbstractPopulationVertex app_vertex:
            The associated application vertex
        :rtype: str
        """
        # Split binary name into title and extension
        name, ext = os.path.splitext(app_vertex.neuron_impl.binary_name)

        # Reunite title and extension and return
        return name + "_neuron" + ext

    @overrides(PopulationMachineCommon._append_additional_provenance)
    def _append_additional_provenance(
            self, provenance_items, prov_list_from_machine, placement):
        # translate into provenance data items
        self._append_neuron_provenance(
            provenance_items, prov_list_from_machine, 0, placement)

    @overrides(PopulationMachineCommon.get_recorded_region_ids)
    def get_recorded_region_ids(self):
        ids = self._app_vertex.neuron_recorder.recorded_ids_by_slice(
            self.vertex_slice)
        return ids

    @inject_items({
        "machine_time_step": "MachineTimeStep",
        "time_scale_factor": "TimeScaleFactor",
        "routing_info": "MemoryRoutingInfos",
        "data_n_time_steps": "DataNTimeSteps",
    })
    @overrides(
        AbstractGeneratesDataSpecification.generate_data_specification,
        additional_arguments={
            "machine_time_step", "time_scale_factor",
            "routing_info", "data_n_time_steps"
        })
    def generate_data_specification(
            self, spec, placement, machine_time_step, time_scale_factor,
            routing_info, data_n_time_steps):
        """
        :param machine_time_step: (injected)
        :param time_scale_factor: (injected)
        :param machine_graph: (injected)
        :param routing_info: (injected)
        :param data_n_time_steps: (injected)
        :param n_key_map: (injected)
        """
        # pylint: disable=arguments-differ
        rec_regions = self._app_vertex.neuron_recorder.get_region_sizes(
            self.vertex_slice, data_n_time_steps)
        self._write_common_data_spec(
            spec, machine_time_step, time_scale_factor, rec_regions)

        self._write_neuron_data_spec(spec, routing_info, machine_time_step)

        # Write information about SDRAM
        n_neurons = self._vertex_slice.n_atoms
        n_synapse_types = self._app_vertex.neuron_impl.get_n_synapse_types()
        spec.reserve_memory_region(
            region=self.REGIONS.SDRAM_EDGE_PARAMS.value,
            size=SDRAM_PARAMS_SIZE, label="SDRAM Params")
        spec.switch_write_focus(self.REGIONS.SDRAM_EDGE_PARAMS.value)
        spec.write_value(
            self.__sdram_partition.get_sdram_base_address_for(self))
        spec.write_value(self.n_bytes_for_transfer)
        spec.write_value(n_neurons)
        spec.write_value(n_synapse_types)
        spec.write_value(len(self.__sdram_partition.pre_vertices))
        spec.write_value(get_n_bits(n_neurons))

        # End the writing of this specification:
        spec.end_specification()

    @inject_items({"routing_info": "MemoryRoutingInfos"})
    @overrides(
        AbstractRewritesDataSpecification.regenerate_data_specification,
        additional_arguments={"routing_info"})
    def regenerate_data_specification(self, spec, placement, routing_info):
        # pylint: disable=too-many-arguments, arguments-differ

        # write the neuron params into the new DSG region
        self._write_neuron_parameters(spec)

        # close spec
        spec.end_specification()

    @overrides(AbstractRewritesDataSpecification.reload_required)
    def reload_required(self):
        return self.__change_requires_neuron_parameters_reload

    @overrides(AbstractRewritesDataSpecification.set_reload_required)
    def set_reload_required(self, new_value):
        self.__change_requires_neuron_parameters_reload = new_value

    @property
    @overrides(ReceivesSynapticInputsOverSDRAM.n_target_neurons)
    def n_target_neurons(self):
        return self._vertex_slice.n_atoms

    @property
    @overrides(ReceivesSynapticInputsOverSDRAM.n_target_synapse_types)
    def n_target_synapse_types(self):
        return self._app_vertex.neuron_impl.get_n_synapse_types()

    @property
    @overrides(ReceivesSynapticInputsOverSDRAM.weight_scales)
    def weight_scales(self):
        machine_timestep = globals_variables.get_simulator().machine_time_step
        return self._app_vertex.get_weight_scales(machine_timestep)

    @property
    @overrides(ReceivesSynapticInputsOverSDRAM.n_bytes_for_transfer)
    def n_bytes_for_transfer(self):
        n_bytes = (2 ** get_n_bits(self.n_target_neurons) *
                   self.n_target_synapse_types * self.N_BYTES_PER_INPUT)
        # May need to add some padding if not a round number of words
        extra_bytes = n_bytes % BYTES_PER_WORD
        if extra_bytes:
            n_bytes += BYTES_PER_WORD - extra_bytes
        return n_bytes

    @overrides(ReceivesSynapticInputsOverSDRAM.sdram_requirement)
    def sdram_requirement(self, sdram_machine_edge):
        if isinstance(sdram_machine_edge.pre_vertex,
                      SendsSynapticInputsOverSDRAM):
            return self.n_bytes_for_transfer
        raise SynapticConfigurationException(
            "Unknown pre vertex type in edge {}".format(sdram_machine_edge))
