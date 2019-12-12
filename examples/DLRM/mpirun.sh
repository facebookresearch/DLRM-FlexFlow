#!/bin/bash

## SLURM scripts have a specific format. 

### Section1: SBATCH directives to specify job configuration

## job name
#SBATCH --job-name=flexflow
## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output=/checkpoint/%u/jobs/flexflow-%j.out
## filename for job standard error output (stderr)
#SBATCH --error=/checkpoint/%u/jobs/flexflow-%j.err

## partition name
#SBATCH --partition=dev
## number of nodes
##SBATCH --nodes=1

## number of tasks per node
##SBATCH --ntasks-per-node=1


### Section 2: Setting environment variables for the job
### Remember that all the module command does is set environment
### variables for the software you need to. Here I am assuming I
### going to run something with python.
### You can also set additional environment variables here and
### SLURM will capture all of them for each task

#export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Debug output
#echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES

# Start clean
module purge

# Load what we need
source ../../FC_env_setup.sh
module load protobuf

### Section 3:
### Run your job. Note that we are not passing any additional
### arguments to srun since we have already specificed the job
### configuration with SBATCH directives
### This is going to run ntasks-per-node x nodes tasks with each
### task seeing all the GPUs on each node. 
#srun -n 2 --label  ./run_random.sh 8 ~/datasets/kaggle_day_1.h5 
#srun --label echo $PROTOBUF
#mpirun -n 16 -N 8 --mca btl_openib_allow_ib 1 --mca btl_openib_warn_default_gid_prefix 0  ./run_random.sh 16 ~/datasets/kaggle_day_1.h5
#./run_random.sh 16 ~/datasets/kaggle_day_1.h5 
per_gpu_batch_size=256
numgpu=8
totalnumgpu=16
batchsize=$((totalnumgpu * per_gpu_batch_size))

./dlrm -ll:gpu ${numgpu} -ll:cpu 1 -ll:fsize 12000 -ll:zsize 20000 -ll:util 1 --arch-sparse-feature-size 64 --arch-embedding-size 1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 --arch-mlp-bot 64-512-512-64 --arch-mlp-top 576-1024-1024-1024-1 --epochs 20 --batch-size ${batchsize} -dm:memorize --strategy ../../src/runtime/dlrm_strategy_${totalnumgpu}gpus.pb
