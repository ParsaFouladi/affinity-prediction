#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --partition=gtx1080
#SBATCH --gres=gpu:4

source activate torch_gpu

python3 gpu_test3.py


