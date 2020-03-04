#!/bin/bash

numgpu="$1"

./batch_matmul_test -ll:gpu ${numgpu} -ll:cpu 4  -ll:util ${numgpu}  -dm:memorize --strategy ../../runtime/dlrm_strategy_emb_1_gpu_${numgpu}_node_1.pb
