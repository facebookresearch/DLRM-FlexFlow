# Copyright 2020 Stanford University
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

GEN_SRC		+= ${FF_HOME}/src/runtime/model.cc ${FF_HOME}/src/mapper/mapper.cc\
		${FF_HOME}/src/runtime/initializer.cc ${FF_HOME}/src/runtime/optimizer.cc\
		${FF_HOME}/src/ops/embedding.cc\
		${FF_HOME}/src/runtime/strategy.pb.cc ${FF_HOME}/src/runtime/strategy.cc\
		${FF_HOME}/src/ops/tests/test_utils.cc #${FF_HOME}/src/ops/embedding_avx2.cc
GEN_GPU_SRC	+= ${FF_HOME}/src/ops/conv_2d.cu ${FF_HOME}/src/runtime/model.cu\
		${FF_HOME}/src/ops/pool_2d.cu ${FF_HOME}/src/ops/batch_norm.cu\
		${FF_HOME}/src/ops/linear.cu ${FF_HOME}/src/ops/softmax.cu\
		${FF_HOME}/src/ops/batch_matmul.cu ${FF_HOME}/src/ops/concat.cu\
		${FF_HOME}/src/ops/flat.cu ${FF_HOME}/src/ops/embedding.cu\
		${FF_HOME}/src/ops/mse_loss.cu ${FF_HOME}/src/ops/transpose.cu\
		${FF_HOME}/src/ops/reshape.cu ${FF_HOME}/src/ops/tanh.cu\
		${FF_HOME}/src/runtime/initializer_kernel.cu ${FF_HOME}/src/runtime/optimizer_kernel.cu\
		${FF_HOME}/src/runtime/accessor_kernel.cu ${FF_HOME}/src/runtime/cuda_helper.cu# .cu files

INC_FLAGS	?= -I${FF_HOME}/include/ -I${FF_HOME}/protobuf/src -I${CUDNN}/include 
LD_FLAGS        ?= -L/usr/local/lib -L${FF_HOME}/protobuf/src/.libs -L${CUDNN}/lib64 -lcudnn -lcublas -lcurand -lprotobuf #-mavx2 -mfma -mf16c
CC_FLAGS	?=
NVCC_FLAGS	?=
GASNET_FLAGS	?=

# For Point and Rect typedefs
CC_FLAGS	+= -std=c++11
NVCC_FLAGS  	+= -std=c++11

ifndef CUDA
#$(error CUDA variable is not defined, aborting build)
endif

ifndef CUDNN
#$(error CUDNN variable is not defined, aborting build)
endif

ifndef LG_RT_DIR
LG_RT_DIR	?= ${FF_HOME}/legion/runtime
endif

ifndef GASNET
GASNET	?= ${FF_HOME}/GASNet-2019.9.0 
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

