#!/bin/bash

numgpu="$1"
python gen_test_case_transpose.py "$2" "$3" "$4"
./transpose_test -ll:gpu ${numgpu} -ll:cpu 4  -ll:util ${numgpu}  -dm:memorize --strategy ../../runtime/dlrm_strategy_emb_1_gpu_${numgpu}_node_1.pb
python validate_transpose.py