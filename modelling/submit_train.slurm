#!/bin/bash



#SBATCH --job-name=train_job           # Job name
#SBATCH --output=train_job_%j.out      # Standard output log
#SBATCH --error=train_job_%j.err       # Standard error log
#SBATCH --nodes=1
#SBATCH --time=03:20:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:2
#SBATCH --mem=20G

source activate torch_gpu

python3 training.py --data_path ../representation/representation_train.h5 --log_file modelling_log

