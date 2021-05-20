/*
 * Copyright (c) 2017-2019 The University of Manchester
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*!
 * \file
 * \brief This file contains the static (non-)implementation of synaptogenesis.
 * \details No functionality is gained with this class.
 */
#include "synaptogenesis_dynamics.h"
#include <debug.h>

bool synaptogenesis_dynamics_initialise(UNUSED address_t sdram_sp_address) {
    return true;
}

bool synaptogenesis_dynamics_rewire(
        UNUSED uint32_t time, UNUSED spike_t *spike,
        UNUSED synaptic_row_t *synaptic_row, UNUSED uint32_t *n_bytes) {
    return false;
}

bool synaptogenesis_row_restructure(
        UNUSED uint32_t time, UNUSED synaptic_row_t row) {
    return false;
}

void synaptogenesis_spike_received(UNUSED uint32_t time, UNUSED spike_t spike) {
}

uint32_t synaptogenesis_n_updates(void) {
    return 0;
}

void print_post_to_pre_entry(void) {
    return;
}
