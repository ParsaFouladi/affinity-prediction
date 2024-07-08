#!/bin/bash

#SBATCH --job-name=docking_job           # Job name
#SBATCH --output=docking_job_%j.out      # Standard output log
#SBATCH --error=docking_job_%j.err       # Standard error log
#SBATCH --ntasks-per-node=40     # Tasks per node
#SBATCH --nodes=2                # Number of nodes requested
#SBATCH --mem=1G                       # Total memory per node
#SBATCH --partition=batch                # Partition name

# Load any necessary modules (if required)
# module load autodock_vina  # Uncomment and adjust if necessary

# Activate the virtual environment (if you use one)
# source /path/to/venv/bin/activate  # Uncomment and adjust if necessary
source activate vina
# Define paths (replace with your actual paths)
RECEPTOR_FOLDER="/mainfs/home/mk6n23/src_parsa/scripts/filter_output/new_alignment_output/test_parsa/"
LIGAND_FOLDER="/mainfs/home/mk6n23/mkdata/mol2_files/new_ligands/pdbqt_version/"
OUTPUT_FOLDER="/mainfs/home/mk6n23/mkdata/docked_complexes/job_test"

# Run the Python script
python3 docking_vina_multiprocessing.py $RECEPTOR_FOLDER $LIGAND_FOLDER $OUTPUT_FOLDER
