# Copyright 2017 Stanford University
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
#

# Put the binary file name here
OUTFILE		?= $(app)
# List all the application source files here
GEN_SRC		?= src/runtime/model.cc src/mapper/mapper.cc src/runtime/initializer.cc src/runtime/optimizer.cc\
		src/ops/embedding_avx2.cc src/ops/embedding.cc src/runtime/strategy.pb.cc src/runtime/strategy.cc\
		src/ops/tests/test_utils.cc src/modules/dot_compressor.cc $(app).cc
GEN_GPU_SRC	?= src/ops/conv_2d.cu src/runtime/model.cu src/ops/pool_2d.cu src/ops/batch_norm.cu src/ops/linear.cu\
		src/ops/softmax.cu src/ops/batch_matmul.cu src/ops/concat.cu src/ops/flat.cu src/ops/embedding.cu src/ops/mse_loss.cu\
		src/runtime/initializer_kernel.cu src/runtime/optimizer_kernel.cu src/runtime/accessor_kernel.cu\
		src/ops/transpose.cu src/ops/reshape.cu src/ops/activations.cu src/runtime/cuda_helper.cu $(app).cu# .cu files

# Flags for directing the runtime makefile what to include
DEBUG           ?= 0		# Include debugging symbols
MAX_DIM         ?= 4		# Maximum number of dimensions
OUTPUT_LEVEL    ?= LEVEL_DEBUG	# Compile time logging level
USE_CUDA        ?= 1		# Include CUDA support (requires CUDA)
USE_GASNET      ?= 1		# Include GASNet support (requires GASNet)
USE_HDF         ?= 1		# Include HDF5 support (requires HDF5)
ALT_MAPPERS     ?= 0		# Include alternative mappers (not recommended)

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	?= -Iinclude/ -I${CUDNN}/include 
LD_FLAGS        ?= -L/usr/local/lib -L${CUDNN}/lib64 -lcudnn -lcublas -lcurand -lprotobuf -mavx2 -mfma -mf16c
CC_FLAGS	?=
NVCC_FLAGS	?=
GASNET_FLAGS	?=

# For Point and Rect typedefs
CC_FLAGS	+= -std=c++11
NVCC_FLAGS  += -std=c++11

ifndef CUDA
#$(error CUDA variable is not defined, aborting build)
endif

ifndef CUDNN
#$(error CUDNN variable is not defined, aborting build)
endif

ifndef LG_RT_DIR
#$(error LG_RT_DIR variable is not defined, aborting build)
LG_RT_DIR	?= legion/runtime
endif

ifndef GASNET
GASNET	?= GASNet-2019.9.0 
endif

ifndef PROTOBUF
#$(error PROTOBUF variable is not defined, aborting build)
endif

PROTOBUF	?= protobuf
INC_FLAGS	+= -I${PROTOBUF}/src
LD_FLAGS	+= -L${PROTOBUF}/src/.libs

ifndef HDF5
HDF5_inc	?= /usr/include/hdf5/serial
HDF5_lib	?= /usr/lib/x86_64-linux-gnu/hdf5/serial
INC_FLAGS	+= -I${HDF5}/
LD_FLAGS	+= -L${HDF5_lib} -lhdf5
endif


###########################################################################
#
#   Don't change anything below here
#   
###########################################################################

include $(LG_RT_DIR)/runtime.mk

