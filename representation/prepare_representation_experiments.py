from representation_functions import *
import os
import sys
import pandas as pd
from mendeleev.fetch import fetch_table
import logging
import re
import h5py
import argparse


#Initialize logger
logging.basicConfig(filename='representation_experiment_log.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s')



# def read_csv_file_train(file_path):
#     correct_column_names = ['PDB_code', 'resolution', 'release_year', '-logKd/Ki', 'Kd/Ki', 'separator', 'reference', 'ligand_name']
#     try:
#         df = pd.read_csv(file_path, delim_whitespace=True, comment='#', skiprows=6, header=None, names=correct_column_names)
#     except Exception as e:
#         logging.error(f"Error reading file: {file_path} - {e}")
#         sys.exit(1)
#     return df

def read_csv_file_test(file_path):
    correct_column_names = ['species', 'EC50_(nM)','kd_lg']
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Error reading file: {file_path} - {e}")
        sys.exit(1)
    return df


def get_binding_affinity_info(species, df):
    try:
        # In order to catch 'Kd=10uM' or 'Ki=10uM' we need to use regex
        # Here we want to catch the binding affinity value and the unit and the type of the binding affinity
        # We are looking for a number followed by a unit (uM, nM, mM, pM, M) and the type of the binding affinity (Ki, Kd)
        
        # First we need to get the binding affinity value
        try:
            p_binding_affinity = float(df[df['species'] == species]['kd_lg'].values[0])
        except:
            logging.error(f"Error getting p_binding_affinity for {species}")
            sys.exit(1)
        # Next we need to get the unit of the binding affinity
        binding_unit = 'nM'
        # Finally we need to get the type of the binding affinity
        binding_type = 'EC50'
        binding_affinity = float(df[df['species'] == species]['EC50_(nM)'].values[0])
        ligand_name = 'cortisol'
    except Exception as e:
        logging.error(f"Error getting binding affinity for {species} - {e}")
        sys.exit(1)
    return binding_affinity, binding_unit, binding_type, p_binding_affinity, ligand_name

def save_representations_to_h5(data, species, binding_information, max_length=400, h5_file_path="representations.h5"):
    """Saves protein-ligand representations and binding affinity to an HDF5 file."""
    with h5py.File(h5_file_path, "a") as f: 
        group_name = f"{species}"
        if group_name in f:
            del f[group_name]  
        group = f.create_group(group_name)
        group.create_dataset("representation", data=data, compression="gzip", compression_opts=9)
        group.attrs["species"] = species
        group.attrs["binding_affinity"] = binding_information['binding_affinity']
        group.attrs["binding_unit"] = binding_information['binding_unit']
        group.attrs["binding_type"] = binding_information['binding_type']
        group.attrs["p_binding_affinity"] = binding_information['p_binding_affinity']
        group.attrs["ligand_name"] = binding_information['ligand_name']

def create_representation(args):
    species, protein_path, ligand_path, df, outpath, max_length, outlier_threshold = args
    try:
        protein_structure = get_protein_structure(protein_path)
        ligand_mol = get_ligand_structure(ligand_path)
        protein_coords = get_amino_acid_cordinates(protein_structure.get_residues())
        lignad_coords = get_ligand_cordinates(ligand_mol)
        number_of_aa = protein_coords.shape[0]

        if outlier_threshold > 1000:
            logging.info(f"Protein {species} has more than 1000 amino acids. Skipping...")
            return None

        if number_of_aa > max_length:
            residues_to_keep = truncation(protein_coords, lignad_coords, max_length=max_length)
            d_matrix = distance_matrix(protein_coords, lignad_coords, protein_size=max_length, residues_to_keep=residues_to_keep)
            w_matrix = molecular_weight(protein_structure, ligand_mol, protein_size=max_length, residues_to_keep=residues_to_keep)
            #v_matrix = vdw_radius_mol(protein_structure, ligand_mol, van_dict, protein_size=max_length, residues_to_keep=residues_to_keep)
            rg_matrix = rg_mol(protein_structure, ligand_mol, protein_size=max_length, residues_to_keep=residues_to_keep)
        elif number_of_aa < max_length:
            residues_to_keep = 'all'
            d_matrix = padding(distance_matrix(protein_coords, lignad_coords, protein_size=number_of_aa, residues_to_keep=residues_to_keep), max_length+1)
            w_matrix = padding(molecular_weight(protein_structure, ligand_mol, protein_size=number_of_aa, residues_to_keep=residues_to_keep), max_length+1)
            #v_matrix = padding(vdw_radius_mol(protein_structure, ligand_mol, van_dict, protein_size=number_of_aa, residues_to_keep=residues_to_keep), max_length+1)
            rg_matrix = padding(rg_mol(protein_structure, ligand_mol, protein_size=number_of_aa, residues_to_keep=residues_to_keep), max_length+1)
        else:
            residues_to_keep = 'all'
            d_matrix = distance_matrix(protein_coords, lignad_coords, protein_size=max_length, residues_to_keep=residues_to_keep)
            w_matrix = molecular_weight(protein_structure, ligand_mol, protein_size=max_length, residues_to_keep=residues_to_keep)
            #v_matrix = vdw_radius_mol(protein_structure, ligand_mol, van_dict, protein_size=max_length, residues_to_keep=residues_to_keep)
            rg_matrix = rg_mol(protein_structure, ligand_mol, protein_size=max_length, residues_to_keep=residues_to_keep)

        stacked_matrix = np.stack((d_matrix, w_matrix, rg_matrix), axis=-1)
        representation = normalize_data(stacked_matrix)
        binding_affinity, binding_unit, binding_type, p_binding_affinity, ligand_name = get_binding_affinity_info(species, df)

        
        save_representations_to_h5(representation, species, {
            'species': species,
            'binding_affinity': binding_affinity,
            'binding_unit': binding_unit,
            'binding_type': binding_type,
            'p_binding_affinity': p_binding_affinity,
            'ligand_name': ligand_name
        }, max_length, outpath)

        return 1
    except Exception as e:
        logging.error(f"Error processing {species}: {e}",exc_info=True)
        return None

def create_representations(data_path, binding_data_path, max_length=400, outpath="representations.h5"):
    df = read_csv_file_test(binding_data_path)
    logging.info("Read the csv file!")

    # Get the list of folders make sure they are not files
    folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    #folders = os.listdir(data_path)
    logging.info("Got the list of folders!")
    logging.info(f"Number of folders: {len(folders)}")
 

    args_list = []
    
    for folder in folders:
        protein_path = os.path.join(data_path, folder, f"{folder}_protein.pdb")
        ligand_path = os.path.join(data_path, folder, f"{folder}_ligand.mol2")
        args = (folder, protein_path, ligand_path, df, outpath, max_length, 1000)
        args_list.append(args)

        create_representation(args)
        logging.info(f"Processed {folder}")

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create protein-ligand representations")
    parser.add_argument('-m', "--max_length", type=int, default=400, help="Maximum length of the protein sequence")
    parser.add_argument("--data_path", type=str, help="Path to the protein-ligand training data")
    parser.add_argument("--binding_data_path", type=str, help="Path to the binding affinity data")
    parser.add_argument('-o', "--output_file", type=str, default="representations_experiment.h5", help="Path to the desired output file")
    args = parser.parse_args()

    if args.data_path and args.binding_data_path:
        create_representations(args.data_path, args.binding_data_path, args.max_length, args.output_file)
    else:
        logging.error("Please provide the data path and binding data path")
        sys.exit(1)
