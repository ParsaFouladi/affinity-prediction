from Bio.PDB import *

# Create a PDB parser object
parser = PDBParser()

# Parse the PDB file
structure = parser.get_structure("protein", "AF-A0A087XJA9-F1-model_v4.pdb")

# Iterate over atoms and retrieve B-factors
red_atoms = []
for model in structure:
    for chain in model:
        for residue in chain:
            for atom in residue:
                # Check if the atom has a B-factor value
                if atom.is_disordered() or atom.bfactor is None:
                    continue
                # Define a threshold for what we consider "red"
                threshold = 50  # Adjust as needed
                if atom.bfactor > threshold:
                    red_atoms.append(atom)

class RedSelect(Select):
    def accept_atom(self, atom):
        if atom in red_atoms:
            return 1
        else:
            return 0

#print(red_atoms)
io = PDBIO()
io.set_structure(structure)
io.save("red_only.pdb", RedSelect())


