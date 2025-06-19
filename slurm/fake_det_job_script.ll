#!/bin/bash

#SBATCH --job-name=qmlpy
#SBATCH --time=96:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=dc162

# Setup environment
module load pytorch/2.2.0-gpu
export XDG_CACHE_HOME=${HOME/home/work}
export MPLCONFIGDIR=${XDG_CACHE_HOME}/.matplotlib
export TORCH_HOME="/work/dc162/dc162/shared/deepfake-detection/models"

# Fix conflict between pytorch own cuda install and local install
unset LD_LIBRARY_PATH

source /work/dc162/dc162/shared/deepfake_venv/bin/activate
cd /work/dc162/dc162/shared/deepfake-detection/

nvidia-smi --loop=60 --filename=/work/dc162/dc162/shared/deepfake-detection/gpu_enabled/slurm/gpu-out-nvidia-smi.out &
srun deepfake_detection.py -c results/Classic_bs32_ep5_hard1k/Classic_bs32_ep5_hard1k.conf.yaml -o
