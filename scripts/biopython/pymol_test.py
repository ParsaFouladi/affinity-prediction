from pymol import cmd
# Load the protein structure file
cmd.load("AF-A0A087XJA9-F1-model_v4.pdb", "protein")

# Find the location of the amino acid "K" in the sequence "KAIEP"
#k_index = cmd.index("KAIEP", "protein")
k_index=cmd.index("pepseq KAIEP")[0][1]
# # test=cmd.select("target","pepseq KAIEP")
# # print(test)
# # Define the start position of the LBD
# print(k_index)
atom=cmd.get_model(selection="id {}".format(k_index)).atom[0]
#print(atom.resi_number)


starting_residue=max(atom.resi_number-7, 1)  # Adjust for 1-based indexing in PyMOL

lbd_start=cmd.index('resi {}'.format(starting_residue))[0][1]

# Define the end position of the LBD
end_flag = False
lbd_end=cmd.count_atoms("protein") + 1
for i in range(lbd_start, cmd.count_atoms("protein") + 1):

    if cmd.select("atom_number","id {} and b<50".format(i)):
        lbd_end = i
        end_flag = True
    if end_flag:
        break

# Create a selection containing the atoms within the defined LBD range
cmd.select("lbd", "id {}-{}".format(lbd_start, lbd_end))

# Save the atoms in the selection to a new PDB file
cmd.save("lbd_structure.pdb", "lbd")
