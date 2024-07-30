from representation_functions import *
import os
import sys
import pandas as pd
from mendeleev.fetch import fetch_table
import logging
import re
import h5py
import argparse
from multiprocessing import Pool, cpu_count, Lock, Manager


#Initialize logger
logging.basicConfig(filename='representation_log.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s')
try:
    ptable = fetch_table("elements")
    cols = ["symbol", "vdw_radius"]
    ptable = ptable[cols]
    ptable.dropna(inplace=True)
    van_dict = ptable.set_index('symbol').T.to_dict('index')['vdw_radius']
    logging.info("Fetched van der Waals radii from Mendeleev database")
except:
    van_dict = {'H': 110.00000000000001, 'He': 140.0, 'Li': 182.0, 'Be': 153.0, 'B': 192.0, 'C': 170.0, 'N': 155.0, 'O': 152.0, 'F': 147.0, 'Ne': 154.0, 'Na': 227.0, 'Mg': 173.0,
                'Al': 184.0, 'Si': 210.0, 'P': 180.0, 'S': 180.0, 'Cl': 175.0, 'Ar': 188.0, 'K': 275.0, 'Ca': 231.0, 'Sc': 215.0, 'Ti': 211.0, 'V': 206.99999999999997, 'Cr': 206.0, 'Mn': 204.99999999999997,
                'Fe': 204.0, 'Co': 200.0, 'Ni': 197.0, 'Cu': 196.0, 'Zn': 200.99999999999997, 'Ga': 187.0, 'Ge': 211.0, 'As': 185.0, 'Se': 190.0, 'Br': 185.0, 'Kr': 202.0, 'Rb': 303.0, 'Sr': 249.00000000000003,
                'Y': 231.99999999999997, 'Zr': 223.0, 'Nb': 218.00000000000003, 'Mo': 217.0, 'Tc': 216.0, 'Ru': 213.0, 'Rh': 210.0, 'Pd': 210.0, 'Ag': 211.0, 'Cd': 218.00000000000003, 'In': 193.0,
                'Sn': 217.0, 'Sb': 206.0, 'Te': 206.0, 'I': 198.0, 'Xe': 216.0, 'Cs': 343.0, 'Ba': 268.0, 'La': 243.00000000000003, 'Ce': 242.0, 'Pr': 240.0, 'Nd': 239.0, 'Pm': 238.0, 'Sm': 236.0, 'Eu': 235.0,
                'Gd': 234.0, 'Tb': 233.0, 'Dy': 231.0, 'Ho': 229.99999999999997, 'Er': 229.0, 'Tm': 227.0, 'Yb': 225.99999999999997, 'Lu': 224.00000000000003,
                'Hf': 223.0, 'Ta': 222.00000000000003, 'W': 218.00000000000003, 'Re': 216.0, 'Os': 216.0, 'Ir': 213.0, 'Pt': 213.0, 'Au': 214.0, 'Hg': 223.0, 'Tl': 196.0, 'Pb': 202.0, 'Bi': 206.99999999999997,
                'Po': 197.0, 'At': 202.0, 'Rn': 220.00000000000003, 'Fr': 348.0, 'Ra': 283.0, 'Ac': 247.00000000000003, 'Th': 245.00000000000003, 'Pa': 243.00000000000003, 'U': 241.0, 'Np': 239.0,
                'Pu': 243.00000000000003, 'Am': 244.0, 'Cm': 245.00000000000003, 'Bk': 244.0, 'Cf': 245.00000000000003, 'Es': 245.00000000000003, 'Fm': 245.00000000000003, 'Md': 246.0, 'No': 246.0, 'Lr': 246.0}

    logging.info("Used default van der Waals radii")

def find_common_folders(dir1, dir2):
    folders1 = [name for name in os.listdir(dir1) if os.path.isdir(os.path.join(dir1, name))]
    folders2 = [name for name in os.listdir(dir2) if os.path.isdir(os.path.join(dir2, name))]
    common_folders = set(folders1).intersection(folders2)
    return common_folders

def read_csv_file_train(file_path):
    correct_column_names = ['PDB_code', 'resolution', 'release_year', '-logKd/Ki', 'Kd/Ki', 'separator', 'reference', 'ligand_name']
    try:
        df = pd.read_csv(file_path, delim_whitespace=True, comment='#', skiprows=6, header=None, names=correct_column_names)
    except Exception as e:
        logging.error(f"Error reading file: {file_path} - {e}")
        sys.exit(1)
    return df

def read_csv_file_test(file_path):
    correct_column_names = ['PDB_code', 'resl', 'year', 'logKa', 'Ka', 'target']
    try:
        df = pd.read_csv(file_path, delim_whitespace=True, comment='#', skiprows=1, header=None, names=correct_column_names)
    except Exception as e:
        logging.error(f"Error reading file: {file_path} - {e}")
        sys.exit(1)
    return df

def common_pdbs(df1, df2):
    common_pdbs = set(df1['PDB_code']).intersection(set(df2['PDB_code']))
    return common_pdbs

def get_binding_affinity_info(pdb_code, df):
    try:
        # In order to catch 'Kd=10uM' or 'Ki=10uM' we need to use regex
        # Here we want to catch the binding affinity value and the unit and the type of the binding affinity
        # We are looking for a number followed by a unit (uM, nM, mM, pM, M) and the type of the binding affinity (Ki, Kd)
        
        # First we need to get the binding affinity value
        binding_affinity = float(re.search(r'(\d+\.?\d*)', df[df['PDB_code'] == pdb_code]['Kd/Ki'].values[0])[0])
        # Next we need to get the unit of the binding affinity
        binding_unit = re.search(r'(uM|nM|mM|pM|M)', df[df['PDB_code'] == pdb_code]['Kd/Ki'].values[0])[0]
        # Finally we need to get the type of the binding affinity
        binding_type = re.search(r'(Ki|Kd)', df[df['PDB_code'] == pdb_code]['Kd/Ki'].values[0])[0]
        resolution = float(df[df['PDB_code'] == pdb_code]['resolution'].values[0])
        p_binding_affinity = float(df[df['PDB_code'] == pdb_code]['-logKd/Ki'].values[0])
        ligand_name = df[df['PDB_code'] == pdb_code]['ligand_name'].values[0]
    except Exception as e:
        logging.error(f"Error getting binding affinity for {pdb_code} - {e}")
        sys.exit(1)
    return binding_affinity, binding_unit, binding_type, resolution, p_binding_affinity, ligand_name

def save_representations_to_h5(data, pdb_code, binding_information, max_length=400, h5_file_path="representations.h5"):
    """Saves protein-ligand representations and binding affinity to an HDF5 file."""
    with h5py.File(h5_file_path, "a") as f: 
        group_name = f"{pdb_code}"
        if group_name in f:
            del f[group_name]  
        group = f.create_group(group_name)
        group.create_dataset("representation", data=data, compression="gzip", compression_opts=9)
        group.attrs["pdb_code"] = pdb_code
        group.attrs["binding_affinity"] = binding_information['binding_affinity']
        group.attrs["binding_unit"] = binding_information['binding_unit']
        group.attrs["binding_type"] = binding_information['binding_type']
        group.attrs["resolution"] = binding_information['resolution']
        group.attrs["p_binding_affinity"] = binding_information['p_binding_affinity']
        group.attrs["ligand_name"] = binding_information['ligand_name']

def create_representation(args):
    pdb_code, protein_path, ligand_path, df, outpath, max_length, outlier_threshold, lock = args
    try:
        protein_structure = get_protein_structure(protein_path)
        ligand_mol = get_ligand_structure(ligand_path)
        protein_coords = get_amino_acid_cordinates(protein_structure.get_residues())
        lignad_coords = get_ligand_cordinates(ligand_mol)
        number_of_aa = protein_coords.shape[0]

        if outlier_threshold > 1000:
            logging.info(f"Protein {pdb_code} has more than 1000 amino acids. Skipping...")
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
            rg_matrix = rg_mol(protein_structure, ligand_mol, protein_size=max_length, residues_to_keep=residues_to_keep)
        else:
            residues_to_keep = 'all'
            d_matrix = distance_matrix(protein_coords, lignad_coords, protein_size=max_length, residues_to_keep=residues_to_keep)
            w_matrix = molecular_weight(protein_structure, ligand_mol, protein_size=max_length, residues_to_keep=residues_to_keep)
            #v_matrix = vdw_radius_mol(protein_structure, ligand_mol, van_dict, protein_size=max_length, residues_to_keep=residues_to_keep)
            rg_matrix = rg_mol(protein_structure, ligand_mol, protein_size=max_length, residues_to_keep=residues_to_keep)

        stacked_matrix = np.stack((d_matrix, w_matrix, rg_matrix), axis=-1)
        representation = normalize_data(stacked_matrix)
        binding_affinity, binding_unit, binding_type, resolution, p_binding_affinity, ligand_name = get_binding_affinity_info(pdb_code, df)

        with lock:
            save_representations_to_h5(representation, pdb_code, {
                'pdb_code': pdb_code,
                'binding_affinity': binding_affinity,
                'binding_unit': binding_unit,
                'binding_type': binding_type,
                'resolution': resolution,
                'p_binding_affinity': p_binding_affinity,
                'ligand_name': ligand_name
            }, max_length, outpath)

        return 1
    except Exception as e:
        logging.error(f"Error processing {pdb_code}: {e}")
        return None

def create_representations(data_path, binding_data_path, binding_data_test, max_length=400, outpath="representations.h5"):
    df = read_csv_file_train(binding_data_path)
    logging.info("Read the csv file!")

    folders = os.listdir(data_path)
    logging.info("Got the list of folders!")
    logging.info(f"Number of folders: {len(folders)}")
    if binding_data_test:
        df_test = read_csv_file_test(binding_data_test)
        common_folders = common_pdbs(df, df_test)
        logging.info(f"Number of common folders: {len(common_folders)}")
    else:
        common_folders = []

    args_list = []
    with Manager() as manager:
        lock = manager.Lock()
        for folder in folders:
            if folder in common_folders:
                logging.info(f"{folder} was found in the test set. Skipping...")
                continue
            protein_path = os.path.join(data_path, folder, f"{folder}_protein.pdb")
            ligand_path = os.path.join(data_path, folder, f"{folder}_ligand.mol2")
            args = (folder, protein_path, ligand_path, df, outpath, max_length, 1000, lock)
            args_list.append(args)

        num_processes = cpu_count()-4

        with Pool(num_processes) as pool:
            for i, result in enumerate(pool.imap_unordered(create_representation, args_list), 1):
                if result is not None:
                    logging.info(f"Representation created for folder {i}/{len(folders)}")
                else:
                    logging.error(f"Error creating representation for folder {i}/{len(folders)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create protein-ligand representations")
    parser.add_argument('-m', "--max_length", type=int, default=400, help="Maximum length of the protein sequence")
    parser.add_argument("--data_path", type=str, help="Path to the protein-ligand training data")
    parser.add_argument("--binding_data_path", type=str, help="Path to the binding affinity data")
    parser.add_argument("--binding_data_test", type=str, default=None, help="Path to the binding affinity data")
    parser.add_argument('-o', "--output_file", type=str, default="representations.h5", help="Path to the desired output file")
    args = parser.parse_args()

    if args.data_path and args.binding_data_path:
        create_representations(args.data_path, args.binding_data_path, args.binding_data_test, args.max_length, args.output_file)
    else:
        logging.error("Please provide the data path and binding data path")
        sys.exit(1)
