#!/bin/bash
#SBATCH -J eval                # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH -n 4                          # Total number of cores requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=128000                   # server memory requested (per node)
#SBATCH -t 960:00:00                   # Time limit (hh:mm:ss)
#SBATCH --partition=gpu               # Request partition
# --constraint="gpu-mid|gpu-high"
# --constraint="[r6000|a5000|a6000|3090|a100|a40]"
#SBATCH --constraint="gpu-mid|gpu-high"
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

#nvidia-smi

# -u makes python unbuffered which will make stdin/stdout write to the capture file more frequently
# JAX_DEBUG_NANS=True XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 python -u "$@"
export TFDS_DATA_DIR=/share/kuleshov/datasets/tensorflow_datasets/ 
JAX_DEFAULT_MATMUL_PRECISION=float32 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 python -u "$@"
