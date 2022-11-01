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

import math
import logging
import struct
import numpy
from pacman.model.resources.constant_sdram import ConstantSDRAM
from spinn_utilities.progress_bar import ProgressBar
from spinn_utilities.log import FormatAdapter
from spynnaker.pyNN.models.common import recording_utils
from pacman.model.resources.variable_sdram import VariableSDRAM
from spinn_front_end_common.utilities.constants import (
    BYTES_PER_WORD, BITS_PER_WORD)
from spynnaker.pyNN.data import SpynnakerDataView
from spynnaker.pyNN.utilities.neo_buffer_database import NeoBufferDatabase

logger = FormatAdapter(logging.getLogger(__name__))
_TWO_WORDS = struct.Struct("<II")


class MultiSpikeRecorder(object):
    __slots__ = [
        "__record"]

    def __init__(self):
        self.__record = False

    @property
    def record(self):
        """
        :rtype: bool
        """
        return self.__record

    @record.setter
    def record(self, record):
        self.__record = record

    def get_sdram_usage_in_bytes(self, n_neurons, spikes_per_timestep):
        """
        :rtype: ~pacman.model.resources.AbstractSDRAM
        """
        if not self.__record:
            return ConstantSDRAM(0)

        out_spike_bytes = (
            int(math.ceil(n_neurons / BITS_PER_WORD)) * BYTES_PER_WORD)
        return VariableSDRAM(0, (2 * BYTES_PER_WORD) + (
            out_spike_bytes * spikes_per_timestep))

    def get_spikes(self, label, region, application_vertex):
        """
        :param str label:
        :param int region:
        :param application_vertex:
        :type application_vertex:
            ~pacman.model.graphs.application.ApplicationVertex
        :return: A numpy array of 2-element arrays of (neuron_id, time)
            ordered by time, one element per event
        :rtype: ~numpy.ndarray(tuple(int,int))
        """
        # pylint: disable=too-many-arguments
        spike_times = list()
        spike_ids = list()

        vertices = application_vertex.machine_vertices
        missing = []
        progress = ProgressBar(
            vertices, "Getting spikes for {}".format(label))
        for vertex in progress.over(vertices):
            placement = SpynnakerDataView.get_placement_of_vertex(vertex)
            vertex_slice = vertex.vertex_slice

            with NeoBufferDatabase() as db:
                times, indices = db.get_multi_spikes(
                    placement.x, placement.y, placement.p, region,
                    SpynnakerDataView.get_simulation_time_step_ms(),
                    vertex_slice, application_vertex.atoms_shape)
            spike_ids.extend(indices)
            spike_times.extend(times)

        if missing:
            logger.warning(
                "Population {} is missing spike data in region {} from the "
                "following cores: {}", label, region,
                recording_utils.make_missing_string(missing))

        if not spike_ids:
            return numpy.zeros((0, 2))

        spike_ids = numpy.hstack(spike_ids)
        spike_times = numpy.hstack(spike_times)
        result = numpy.dstack((spike_ids, spike_times))[0]
        return result[numpy.lexsort((spike_times, spike_ids))]
