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
import logging
from spynnaker.pyNN.models.neuron.plasticity.stdp.timing_dependence import (
    TimingDependenceRecurrent as
    _BaseClass)

_defaults = _BaseClass.default_parameters
logger = logging.getLogger(__name__)


class TimingDependenceRecurrent(_BaseClass):
    """
    .. deprecated:: 6.0
        Use
        :py:class:`spynnaker.pyNN.models.neuron.plasticity.stdp.timing_dependence.TimingDependenceRecurrent`
        instead.
    """
    __slots__ = []

    def __init__(
            self, accumulator_depression=_defaults['accumulator_depression'],
            accumulator_potentiation=_defaults['accumulator_potentiation'],
            mean_pre_window=_defaults['mean_pre_window'],
            mean_post_window=_defaults['mean_post_window'],
            dual_fsm=_defaults['dual_fsm'], A_plus=0.01, A_minus=0.01):
        """
        :param int accumulator_depression:
        :param int accumulator_potentiation:
        :param float mean_pre_window:
        :param float mean_post_window:
        :param bool dual_fsm:
        :param float A_plus: :math:`A^+`
        :param float A_minus: :math:`A^-`
        """
        # pylint: disable=too-many-arguments
        super(TimingDependenceRecurrent, self).__init__(
            accumulator_depression=accumulator_depression,
            accumulator_potentiation=accumulator_potentiation,
            mean_pre_window=mean_pre_window,
            mean_post_window=mean_post_window,
            dual_fsm=dual_fsm, A_plus=A_plus, A_minus=A_minus)
        logger.warning(
            "please use spynnaker.pyNN.models.neuron.plasticity.stdp."
            "timing_dependence.TimingDependenceRecurrent instead")
