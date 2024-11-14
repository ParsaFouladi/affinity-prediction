#!/bin/bash

#SBATCH --job-name=docking_job           # Job name
#SBATCH --output=docking_job_%j.out      # Standard output log
#SBATCH --error=docking_job_%j.err       # Standard error log
#SBATCH --ntasks-per-node=40     # Tasks per node
#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --mem=1G                       # Total memory per node
#SBATCH --partition=batch                # Partition name
#SBATCH --time=06:00:00

# Load any necessary modules (if required)
# module load autodock_vina  # Uncomment and adjust if necessary

# Activate the virtual environment (if you use one)
# source /path/to/venv/bin/activate  # Uncomment and adjust if necessary
source activate vina
# Define paths (replace with your actual paths)
#RECEPTOR_FOLDER="/mainfs/home/mk6n23/new_enviroment_parsa/data/mr_proteins_filtered/"
#LIGAND_FOLDER="/mainfs/home/mk6n23/mkdata/mol2_files/new_ligands/pdbqt_version/"
#OUTPUT_FOLDER="/mainfs/home/mk6n23/mkdata/docked_complexes/mr_results2"

# Run the Python script
python3 docking_vina_multiprocessing.py --receptor_folder /mainfs/home/mk6n23/new_enviroment_parsa/data/ten_proteins/ --ligand_folder /mainfs/home/mk6n23/new_enviroment_parsa/data/cortisol/ --output_folder /mainfs/home/mk6n23/new_enviroment_parsa/data/results/sheet2_10_e128_5/ -e 128
