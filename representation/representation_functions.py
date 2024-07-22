import numpy as np
from Bio.PDB import *
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.spatial.distance import cdist
from mendeleev import element
from mendeleev.fetch import fetch_table
from sklearn.preprocessing import StandardScaler
import warnings
import scipy.sparse as sp

# Filter out specific PDBConstructionWarnings
warnings.filterwarnings("ignore", message="Ignoring unrecognized record 'END'")
warnings.filterwarnings("ignore", message="Ignoring unrecognized record 'TER'")

def get_protein_structure(protein_path):
    """
    Parse and return the protein structure from a PDB file.

    Args:
        protein_path (str): Path to the PDB file.

    Returns:
        structure: Parsed protein structure.
    """
    parser = PDBParser()
    structure = parser.get_structure("protein", f"{protein_path}")
    return structure

def get_ligand_structure(ligand_path):
    """
    Parse and return the ligand structure from a Mol2 file.

    Args:
        ligand_path (str): Path to the Mol2 file.

    Returns:
        ligand_mol: Parsed ligand molecule.
    """
    #ligand_mol = Chem.MolFromMol2File(ligand_path)
    ligand_mol=Chem.SDMolSupplier(ligand_path)
    return ligand_mol

def get_amino_acid_cordinates(structure_residues):
    """
    Get the geometric center coordinates of amino acids from the structure residues.

    Args:
        structure_residues (list): List of residues in the protein structure.

    Returns:
        np.ndarray: Numpy array of geometric center coordinates of amino acids.
    """
    aa_coords_geo = [residue.center_of_mass(geometric=True) for residue in structure_residues if residue.get_resname() != 'HOH']
    aa_coords_geo = np.array(aa_coords_geo)
    return aa_coords_geo

def get_ligand_cordinates(ligand_mol):
    """
    Get the geometric center coordinates of the ligand molecule.

    Args:
        ligand_mol: Ligand molecule.

    Returns:
        np.ndarray: Numpy array of geometric center coordinates of the ligand.
    """
    # Iterate over the molecules in the SDF file
    for mol in ligand_mol:
      if mol is None:
        continue  # Skip any molecules that couldn't be parsed
    
    # Get the conformer of the molecule
      conf = mol.GetConformer()
    #conf = ligand_mol.GetConformer()
    ligand_coords = conf.GetPositions()
    ligand_coord = ligand_coords.mean(axis=0)
    return ligand_coord

def distance_function(structure1_coords, structure2_coords):
    """
    Compute the pairwise distances between two sets of coordinates.

    Args:
        structure1_coords (np.ndarray): Numpy array of coordinates for the first structure.
        structure2_coords (np.ndarray): Numpy array of coordinates for the second structure.

    Returns:
        np.ndarray: Distance matrix.
    """
    distances_geo = cdist(structure1_coords, structure2_coords)
    return distances_geo

def padding(numpy_array, max_length):
    """
    Pad or truncate a numpy array to a specified maximum length.

    Args:
        numpy_array (np.ndarray): Numpy array to pad or truncate.
        max_length (int): Maximum length for the array.

    Returns:
        np.ndarray: Padded or truncated numpy array.
    """
    padded_array = np.zeros((max_length, max_length))
    num_pad = max_length - len(numpy_array)
    pad_left = num_pad // 2
    pad_right = num_pad - pad_left
    numpy_array = np.pad(numpy_array, (pad_left, pad_right), mode='constant', constant_values=0)
    return numpy_array

def truncation(protein_coords, ligand_coords, max_length):
    """
    Truncate the protein coordinates to keep only those within a specified distance from the ligand coordinates.

    Logic:
    1. Compute the pairwise distances between the protein residues and the ligand.
    2. Sort the distances and keep the indices of the residues that are closest to the ligand.
    3. Return the indices of the residues to keep.

    Args:
        protein_coords (np.ndarray): Numpy array of protein coordinates.
        ligand_coords (np.ndarray): Numpy array of ligand coordinates.
        max_length (int): Maximum number of residues to keep.

    Returns:
        np.ndarray: Indices of residues to keep.
    """
    distances = distance_function(protein_coords, [ligand_coords])
    sorted_indices = distances.argsort(axis=0).flatten()
    residues_to_keep = sorted_indices[:max_length]
    residues_to_keep.sort()
    return residues_to_keep

def distance_matrix(protein_coords, ligand_coord, protein_size, residues_to_keep='all'):
    """
    Compute the distance matrix including both protein residues and the ligand.

    Logic:
    1. Keep only the coordinates of the residues to keep.
    2. Compute the pairwise distances between the protein residues and the ligand.

    Args:
        protein_coords (np.ndarray): Numpy array of protein coordinates.
        ligand_coord (np.ndarray): Numpy array of ligand coordinates.
        protein_size (int): Size of the protein.
        residues_to_keep (str or list): List of residue indices to keep or 'all' to keep all.

    Returns:
        np.ndarray: Distance matrix.
    """
    if isinstance(residues_to_keep, str) and residues_to_keep == 'all':
        residues_to_keep = range(protein_size)
    
    aa_coords_to_keep = [protein_coords[i] for i in residues_to_keep]
    all_coordinates = aa_coords_to_keep + [ligand_coord]
    distances_matrix = distance_function(all_coordinates, all_coordinates)
    return distances_matrix

def molecular_weight(structure, ligand_mol, protein_size, residues_to_keep='all'):
    """
    Compute the molecular weight matrix including both protein residues and the ligand.

    Args:
        structure: Protein structure.
        ligand_mol: Ligand molecule.
        protein_size (int): Size of the protein.
        residues_to_keep (str or list): List of residue indices to keep or 'all' to keep all.

    Returns:
        np.ndarray: Molecular weight matrix.
    """
    structure_residues = structure.get_residues()
    matrix_length = protein_size + 1
    aa_masses = []

    if isinstance(residues_to_keep, str) and residues_to_keep == 'all':
        residues_to_keep = range(protein_size)
    
    residue_number = 0
    for residue in structure_residues:
        if residue.id[0] in [" ", "H"] and residue_number in residues_to_keep:
            aa_masses.append(ProteinAnalysis(residue.get_resname()).molecular_weight())
        residue_number += 1

    ligand_weight = Descriptors.MolWt(ligand_mol)
    final_lst = aa_masses + [ligand_weight]
    mass_channel = np.zeros((matrix_length, matrix_length))
    np.fill_diagonal(mass_channel, final_lst)
    return mass_channel

def vdw_radius_mol(structure, ligand_mol, van_dict, protein_size, residues_to_keep='all'):
    """
    Compute the van der Waals radius matrix including both protein residues and the ligand and save the mean.

    Args:
        structure: Protein structure.
        ligand_mol: Ligand molecule.
        van_dict (dict): Dictionary of van der Waals radii.
        protein_size (int): Size of the protein.
        residues_to_keep (str or list): List of residue indices to keep or 'all' to keep all.

    Returns:
        np.ndarray: Van der Waals radius matrix.
    """
    structure_residues = structure.get_residues()
    matrix_length = protein_size + 1
    aa_vdw_radii = []

    if isinstance(residues_to_keep, str) and residues_to_keep == 'all':
        residues_to_keep = range(protein_size)
    
    residue_number = 0
    for residue in structure_residues:
        if residue.id[0] in [" ", "H"] and residue_number in residues_to_keep:
            atoms = [atom.element for atom in residue.get_atoms()]
            radii = [van_dict[atom] for atom in atoms]
            aa_vdw_radii.append(np.mean(radii))  # Average radii
        residue_number += 1
    
    ligand_vdw_radius = []
    for atom in ligand_mol.GetAtoms():
        atomic_symbol = atom.GetSymbol()
        radius = van_dict[atomic_symbol]
        ligand_vdw_radius.append(radius)
    
    ligand_vdw_radius = np.mean(ligand_vdw_radius)
    final_lst = aa_vdw_radii + [ligand_vdw_radius]
    radius_channel = np.zeros((matrix_length, matrix_length))
    np.fill_diagonal(radius_channel, final_lst)
    return radius_channel

def stack_all(d_matrix, w_matrix, v_radius):
    """
    Stack the distance matrix, molecular weight matrix, and van der Waals radius matrix.

    Args:
        d_matrix (np.ndarray): Distance matrix.
        w_matrix (np.ndarray): Molecular weight matrix.
        v_radius (np.ndarray): Van der Waals radius matrix.

    Returns:
        np.ndarray: Stacked matrix.
    """
    # Convert diagonal channels to sparse matrices (CSR format)
    w_matrix = sp.csr_matrix(w_matrix)
    v_radius = sp.csr_matrix(v_radius)

    return np.stack([d_matrix, w_matrix, v_radius], axis=-1)

def normalize_data(stacked_array):
    """
    Normalize the data in the stacked array. Notice that for the last two channels (molecular weight and van der Waals radius), we normalize only the diagonal.


    Args:
        stacked_array (np.ndarray): Stacked array containing distance, molecular weight, and van der Waals radius matrices.

    Returns:
        np.ndarray: Normalized stacked array.
    """
    all_channels = stacked_array.copy()
    for i in range(all_channels.shape[-1]):  # Iterate over channels
        # Create Mask (Assuming 0 is used for padding)
        mask = all_channels[:, :, i] != 0

        # Apply StandardScaler (or any other normalization method)
        scaler = StandardScaler()
        all_channels[:, :, i][mask] = scaler.fit_transform(all_channels[:, :, i][mask].reshape(-1, 1)).flatten()

        # if i == 0:
        #     scaler = StandardScaler()
        #     all_channels[:, :, i] = scaler.fit_transform(all_channels[:, :, i])
        # else:
        #     diag_values = all_channels[:, :, i].diagonal()  # Extract diagonal
        #     scaler = StandardScaler()
        #     normalized_diag = scaler.fit_transform(diag_values.reshape(-1, 1)).flatten()  # Normalize
        #     np.fill_diagonal(all_channels[:, :, i], normalized_diag)
    # convert to float32
    all_channels = all_channels.astype(np.float32)
    return all_channels
