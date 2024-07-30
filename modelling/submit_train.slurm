#!/bin/bash



#SBATCH --job-name=train_job           # Job name
#SBATCH --output=train_job_%j.out      # Standard output log
#SBATCH --error=train_job_%j.err       # Standard error log
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=20G

source activate torch_gpu

python3 training.py --data_path /mainfs/scratch/mk6n23/parsa_data/representations/representation_train_300_van.h5 --log_file modelling_log_300_van_deep --epochs 20 --lr 0.001 --batch_size 64 --val_split 0.05 --input_channels 3 --height 301 --width 301 -o model_300_van_deep

