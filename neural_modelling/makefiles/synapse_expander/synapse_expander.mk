# Copyright (c) 2017 The University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

APP = synapse_expander
BUILD_DIR = build/
SOURCES = synapse_expander/rng.c \
          synapse_expander/common_kernel.c \
          synapse_expander/param_generator.c \
          synapse_expander/connection_generator.c \
          synapse_expander/matrix_generator.c \
          synapse_expander/synapse_expander.c \
          neuron/population_table/population_table_binary_search_impl.c

include ../neural_support.mk
