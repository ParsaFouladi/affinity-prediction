#!/bin/bash



#SBATCH --job-name=train_job           # Job name
#SBATCH --output=train_job_%j.out      # Standard output log
#SBATCH --error=train_job_%j.err       # Standard error log
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=20G

source activate torch_gpu

python3 training.py --data_path /mainfs/scratch/mk6n23/parsa_data/representations/representation_train_300_rg_pad.h5 --log_file text_logs/modelling_log_300_rg_resnet101_epoch120_customInit --epochs 120 --lr 0.001 --batch_size 32 --val_split 0.05 -c 3 --height 301 -w 301 -o model_300_rg_resnet101_epoch120 --model_type ResNet101

