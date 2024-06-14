from vina import Vina
import sys
import os
from multiprocessing import Pool, cpu_count
import logging

# Configure logging to write to a text file
logging.basicConfig(filename='docking_log.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s')

# Define paths to the folders containing receptors and ligands
receptor_folder = sys.argv[1]
ligand_folder = sys.argv[2]
output_folder = sys.argv[3]

# Get a list of receptor and ligand files
receptor_files = [os.path.join(receptor_folder, file) for file in os.listdir(receptor_folder) if file.endswith('.pdbqt')]
ligand_files = [os.path.join(ligand_folder, file) for file in os.listdir(ligand_folder) if file.endswith('.pdbqt')]

# Remove spaces from the file names
for i in range(len(receptor_files)):
    new_name = receptor_files[i].replace(" ", "_")
    os.rename(receptor_files[i], new_name)
    receptor_files[i] = new_name
for i in range(len(ligand_files)):
    new_name = ligand_files[i].replace(" ", "_")
    os.rename(ligand_files[i], new_name)
    ligand_files[i] = new_name

# Print number of total pairs which is number of receptors * number of ligands
total_pairs = len(receptor_files) * len(ligand_files)
logging.info(f"Number of total pairs: {total_pairs}")

def dock_pair(pair):
    receptor_file, ligand_file, output_folder = pair
    try:
        receptor_name = os.path.splitext(os.path.basename(receptor_file))[0]
        ligand_name = os.path.splitext(os.path.basename(ligand_file))[0]

        v = Vina(sf_name='vina')
        v.set_receptor(receptor_file)
        v.set_ligand_from_file(ligand_file)
        v.compute_vina_maps(center=[5, 2, -15], box_size=[35, 35, 35])

        v.dock(exhaustiveness=32, n_poses=20)
        output_filename = '{}_{}_vina_out.pdbqt'.format(receptor_name, ligand_name)
        output_filepath = os.path.join(output_folder, output_filename)
        v.write_poses(output_filepath, n_poses=1, overwrite=True)

        logging.info(f"Docking completed: {receptor_name} with {ligand_name}")
        return receptor_file, ligand_file
    except Exception as e:
        logging.error(f"Error docking {receptor_file} with {ligand_file}: {e}")
        return None

if __name__ == "__main__":
    pairs = [(receptor_file, ligand_file, output_folder) for receptor_file in receptor_files for ligand_file in ligand_files]
    num_processes = cpu_count()
    number_of_docked_pairs = 0

    with Pool(num_processes) as pool:
        for i, result in enumerate(pool.imap_unordered(dock_pair, pairs), 1):
            if result:
                number_of_docked_pairs += 1
                logging.info(f'Docking pair {number_of_docked_pairs} of {total_pairs} is finished.')

    logging.info('All docking tasks are complete.')
