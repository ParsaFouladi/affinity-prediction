from vina import Vina
import sys
import os
from multiprocessing import Pool, cpu_count
import logging
import argparse
import datetime


def dock_pair(pair):
    receptor_file, ligand_file, output_folder,exhaustiveness,centre,box_size = pair
    try:
        receptor_name = os.path.splitext(os.path.basename(receptor_file))[0]
        ligand_name = os.path.splitext(os.path.basename(ligand_file))[0]

        v = Vina(sf_name='vina')
        v.set_receptor(receptor_file)
        v.set_ligand_from_file(ligand_file)
        # v.compute_vina_maps(center=[5, 2, -15], box_size=[20, 20, 20])
        v.compute_vina_maps(center=centre, box_size=box_size)
        
        v.dock(exhaustiveness=exhaustiveness, n_poses=20)
        output_filename = '{}_{}_vina_out.pdbqt'.format(receptor_name, ligand_name)
        output_filepath = os.path.join(output_folder, output_filename)
        v.write_poses(output_filepath, n_poses=1, overwrite=True)

        logging.info(f"Docking completed: {receptor_name} with {ligand_name}")
        return receptor_file, ligand_file
    except Exception as e:
        logging.error(f"Error docking {receptor_file} with {ligand_file}: {e}")
        return None

def main(args):
    # Configure logging to write to a text file
    current_date = datetime.datetime.now().strftime("%d%m%Y")
    logging.basicConfig(filename=f'docking_log_{current_date}.log', level=logging.INFO, 
                        format='%(asctime)s %(levelname)s:%(message)s')



    # Define paths to the folders containing receptors and ligands
    receptor_folder = args.receptor_folder
    ligand_folder = args.ligand_folder
    output_folder = args.output_folder
    exhaustiveness = args.exhaustiveness
    centre=args.center
    box_size=args.box_size
    num_cpus=args.num_cpus

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

    pairs = [(receptor_file, ligand_file, output_folder,exhaustiveness,centre,box_size) for receptor_file in receptor_files for ligand_file in ligand_files]
    num_processes = num_cpus
    number_of_docked_pairs = 0

    logging.info(f"Total number of CPUs available in this run: {num_processes}")

    with Pool(num_processes) as pool:
        for i, result in enumerate(pool.imap_unordered(dock_pair, pairs), 1):
            if result:
                number_of_docked_pairs += 1
                logging.info(f'Docking pair {number_of_docked_pairs} of {total_pairs} is finished.')

    logging.info('All docking tasks are complete.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create protein-ligand representations")
    parser.add_argument("--receptor_folder", type=str,help="Path to the folder containing receptor files")
    parser.add_argument("--ligand_folder", type=str, help="Path to the folder containing ligand files")
    parser.add_argument("--output_folder", type=str, help="Path to the output folder")
    # exhaustiveness
    parser.add_argument("--exhaustiveness", "-e",type=int, default=32, help="Exhaustiveness parameter for Vina")

    # add the arguments for centre coordinates and box size
    parser.add_argument("--center", "-c", type=float, nargs=3, default=[5, 2, -15], help="Center coordinates for Vina")
    parser.add_argument("--box_size", "-b", type=float, nargs=3, default=[20, 20, 20], help="Box size for Vina")
    
    # number of CPUs to use
    parser.add_argument("--num_cpus", "-n", type=int, default=cpu_count(), help="Number of CPUs to use")
    args = parser.parse_args()

    main(args)
    
