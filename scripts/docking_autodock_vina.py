from vina import Vina
import sys
import os

# Define paths to the folders containing receptors and ligands
# receptor_folder = '/mainfs/home/mk6n23/src_parsa/scripts/filter_output/new_alignment_output/new_alignment_output/pdbqt_version/'
# ligand_folder = '/mainfs/home/mk6n23/mkdata/mol2_files/new_ligands/pdbqt_version/'
# output_folder = '/mainfs/home/mk6n23/mkdata/docked_complexes/new_parsa_set/'

receptor_folder = sys.argv[1]
ligand_folder = sys.argv[2]
output_folder = sys.argv[3]

# Get a list of receptor and ligand files
receptor_files = [os.path.join(receptor_folder, file) for file in os.listdir(receptor_folder) if file.endswith('.pdbqt')]
ligand_files = [os.path.join(ligand_folder, file) for file in os.listdir(ligand_folder) if file.endswith('.pdbqt')]

#remove the spaces from the file names
for i in range(len(receptor_files)):
    os.rename(receptor_files[i], receptor_files[i].replace(" ", "_"))
    receptor_files[i] = receptor_files[i].replace(" ", "_")
for i in range(len(ligand_files)):
    os.rename(ligand_files[i], ligand_files[i].replace(" ", "_"))
    ligand_files[i] = ligand_files[i].replace(" ", "_")

#print number of total pairs which is number of receptors * number of ligands
print("Number of total pairs: ", len(receptor_files)*len(ligand_files))


# Iterate over each combination of receptor and ligand
#number of docked pairs
number_of_docked_pairs = 0
for receptor_file in receptor_files:
    for ligand_file in ligand_files:
        number_of_docked_pairs += 1

        print('Docking pair %d of %d is started.' % (number_of_docked_pairs, len(receptor_files)*len(ligand_files)))
        # Extract receptor and ligand names
        receptor_name = os.path.splitext(os.path.basename(receptor_file))[0]
        ligand_name = os.path.splitext(os.path.basename(ligand_file))[0]

        # Instantiate Vina object
        v = Vina(sf_name='vina')

        # Set receptor and ligand
        v.set_receptor(receptor_file)
        v.set_ligand_from_file(ligand_file)

        # Compute Vina maps
        v.compute_vina_maps(center=[5, 2, -15], box_size=[35, 35, 35]) #change grid values

        # Score the current pose
        # energy = v.score()
        # print('Score before minimization: %.3f (kcal/mol)' % energy[0])

        # Dock the ligand
        v.dock(exhaustiveness=32, n_poses=20)
        
        # Save docked files to the output folder
        output_filename = '{}_{}_vina_out.pdbqt'.format(receptor_name, ligand_name)
        output_filepath = os.path.join(output_folder, output_filename)
        v.write_poses(output_filepath, n_poses=1, overwrite=True)
        print('Docking pair %d of %d is finished.' % (number_of_docked_pairs, len(receptor_files)*len(ligand_files)))
        
