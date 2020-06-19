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
nnodes=$1

per_gpu_batch_size=512
pngpu=8
ngpu=$((nnodes * pngpu))
nemb=128
batchsize=$((ngpu * per_gpu_batch_size))
pnemb=$((nemb / nnodes))

embsize="10000"
embsizes=$embsize
sparse_feature_size=256
first_top_mlp_size=$((nemb * sparse_feature_size + sparse_feature_size))

for ((a=1 ; a<$nemb ; a++)) 
do
	embsizes+="-"
	embsizes+=$embsize
done
echo $embsizes

strategy_file=../../src/runtime/dlrm_strategy_emb_${pnemb}_gpu_${pngpu}_node_${nnodes}.pb
echo $strategy_file

#GASNET_FREEZE_ON_ERROR=1 ./dlrm -ll:gpu ${pngpu} -ll:cpu 1 -ll:dma 1 -ll:ahandlers 1 -ll:fsize 12000 -ll:zsize 20000 -ll:util 4 -lg:prof 2 -lg:prof_logfile prof_%.gz --nodes ${nnodes} --arch-sparse-feature-size 64 --arch-embedding-size ${embsizes} --arch-mlp-bot 64-512-512-64 --arch-mlp-top 576-1024-1024-1024-1 --epochs 5 --batch-size ${batchsize} -dm:memorize --strategy ${strategy_file}

./dlrm -ll:gpu ${pngpu} -ll:cpu 1 -ll:dma 4 -ll:ahandlers 2 -ll:fsize 12000 -ll:zsize 20000 -ll:util 8 --nodes ${nnodes} --arch-sparse-feature-size ${sparse_feature_size} --arch-embedding-size ${embsizes} --arch-mlp-bot 64-1024-1024-1024-${sparse_feature_size} --arch-mlp-top ${first_top_mlp_size}-1024-1024-1024-1024-1 --epochs 5 --batch-size ${batchsize} --strategy ${strategy_file}
#./dlrm -ll:gpu ${pngpu} -ll:cpu 1 -ll:dma 4 -ll:ahandlers 2  -ll:fsize 12000 -ll:zsize 20000 -ll:util 8 --nodes ${nnodes}  --embedding-bag-size 100 --arch-sparse-feature-size 64 --arch-embedding-size ${embsizes} --arch-mlp-bot 2048-4096-1024-64 --arch-mlp-top 8265-1024-1024-1 --epochs 15 --batch-size ${batchsize} --strategy ${strategy_file} 
#-dm:memoize  
