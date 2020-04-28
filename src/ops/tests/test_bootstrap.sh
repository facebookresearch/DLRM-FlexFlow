#!/bin/bash
# request 8 gpus resource for testing
numgpu="$1"
numnode="$2"
lifetime="$3"
srun --nodes=${numnode} --gres=gpu:${numgpu} --cpus-per-task=80 --partition=dev --time=${lifetime} --pty /bin/bash -l