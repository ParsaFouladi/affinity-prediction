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


def get_amino_acid_cordinates(structure_residues):
  aa_coords_geo = [residue.center_of_mass(geometric=True) for residue in structure_residues if residue.get_resname()!='HOH']

  #Convert it to numpy array
  aa_coords_geo = np.array(aa_coords_geo)

  return aa_coords_geo

def get_ligand_cordinates(ligand_mol):
  #ligand_coord = mol2.GetConformer().GetPositions().mean(axis=0)
  conf = ligand_mol.GetConformer()
  ligand_coords = conf.GetPositions()
  ligand_coord = ligand_coords.mean(axis=0) 
  return ligand_coord

def distance_function(structure1_coords,structure2_coords):
  distances_geo = cdist(structure1_coords, structure2_coords)
  # Sort residues by distance (ascending)
  #sorted_indices_geo = distances_geo.argsort(axis=0).flatten()
  return distances_geo

def padding(numpy_array, max_length):
  padded_array = np.zeros((max_length, max_length))
  # Symmetric Padding or Truncation

  num_pad = max_length - len(numpy_array)
  pad_left = num_pad // 2
  pad_right = num_pad - pad_left

  numpy_array = np.pad(numpy_array, (pad_left, pad_right), mode='constant', constant_values=0)
  return numpy_array

def truncation(protein_coords,ligand_coords,max_length):
  distances = distance_function(protein_coords, [ligand_coords])

  # Sort residues by distance (ascending)
  sorted_indices = distances.argsort(axis=0).flatten()

  residues_to_keep = sorted_indices[:max_length]
  #residues_to_remove = sorted_indices[max_length:]

  residues_to_keep.sort()
  return residues_to_keep

def distance_matrix(protein_coords,ligand_coord,max_length,residues_to_keep='all'):
  matrix_size=max_length+1
  if isinstance(residues_to_keep, str) and residues_to_keep=='all':
    residues_to_keep=range(matrix_size)
  
  aa_coords_to_keep = [protein_coords[i] for i in residues_to_keep]
  all_cordinates=aa_coords_to_keep+[ligand_coord]
  distances_matrix = distance_function(all_cordinates, all_cordinates)

  return distances_matrix


def molecular_weight(structure_residues,ligand_mol,max_length,residues_to_keep='all'):
  matrix_length=max_length+1

  aa_masses = []
  if isinstance(residues_to_keep, str) and residues_to_keep=='all':
    residues_to_keep=range(matrix_length)
  residue_number=0
  for residue in structure_residues:
    if residue.get_resname() !='HOH' and residue_number in residues_to_keep:
        aa_masses.append(ProteinAnalysis(residue.get_resname()).molecular_weight())
    residue_number+=1

  ligand_weight=Descriptors.MolWt(ligand_mol)

  final_lst=aa_masses + [ligand_weight]


  mass_channel = np.zeros((matrix_length, matrix_length))
  

  np.fill_diagonal(mass_channel, final_lst)
  

  return mass_channel

def vdw_radius_mol(structure_residues,ligand_mol,van_dict,max_length,residues_to_keep='all'):
  matrix_length=max_length+1
  aa_vdw_radii = []
  residue_number=0
  for residue in structure_residues :
    if residue.get_resname() !='HOH' and residue_number in residues_to_keep:
        atoms = [atom.element for atom in residue.get_atoms()]
        #print(atoms)
        radii = [van_dict[atom] for atom in atoms]
        #print(radii)
        aa_vdw_radii.append(np.mean(radii))  # Average radii
    
    residue_number+=1
  
  ligand_vdw_radius = []

  for atom in ligand_mol.GetAtoms():
      atomic_symbol = atom.GetSymbol()
      radius = van_dict[atomic_symbol] #if atomic_symbol in van_dict else 0
      ligand_vdw_radius.append(radius)

  ligand_vdw_radius = np.mean(ligand_vdw_radius) 

  final_lst=aa_vdw_radii + [ligand_vdw_radius]
  radius_channel = np.zeros((matrix_length, matrix_length))

  np.fill_diagonal(radius_channel, final_lst)

  return (radius_channel)

def stack_all(d_matrix,w_matrx,v_radius):
  return np.stack([d_matrix,w_matrx,v_radius],axis=-1)


def normalize_data(stacked_array):
  all_channels=stacked_array.copy()
# Normalize each channel separately
  for i in range(all_channels.shape[-1]):  # Iterate over channels
    scaler = StandardScaler()
    all_channels[:, :, i] = scaler.fit_transform(all_channels[:, :, i])
  return all_channels