#!/bin/bash

#SBATCH --job-name=representation_job           # Job name
#SBATCH --output=representation_job_%j.out      # Standard output log
#SBATCH --error=representation_job_%j.err       # Standard error log
#SBATCH --ntasks-per-node=40     # Tasks per node
#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --mem=10G                       # Total memory per node
#SBATCH --partition=batch                # Partition name

# Load any necessary modules (if required)
# module load autodock_vina  # Uncomment and adjust if necessary

# Activate the virtual environment (if you use one)
# source /path/to/venv/bin/activate  # Uncomment and adjust if necessary
source activate bio


# Run the Python script
python3 prepare_representation.py --data_path /mainfs/scratch/mk6n23/parsa_data/pdbbind_train/refined-set/ --binding_data_path /mainfs/scratch/mk6n23/parsa_data/pdbbind_train/refined_data.2020 --binding_data_test /mainfs/scratch/mk6n23/parsa_data/pdbbind_train/CoreSet.dat -o representation_train.h5
