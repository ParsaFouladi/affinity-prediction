from Bio.PDB import *

# Create a PDB parser object
parser = PDBParser()

# Parse the PDB file
structure = parser.get_structure("protein", "data\AF-A0A087XJA9-F1-model_v4.pdb")
ppb = PPBuilder()
pep=ppb.build_peptides(structure)
seq = pep[0].get_sequence()

k_index=seq.find("KAIEP")
starting_index=k_index-7

class RedSelect(Select):
    def __init__(self):
        self.terminate = False

    # Accept the risudes with index of higher than starting index and atoms with bfactor higher than threshold
    def accept_residue(self, residue):
        
        if residue.get_id()[1] > starting_index and not self.terminate:
            #get all the atoms in the residue
            atoms = residue.get_list()
            for atom in atoms:
                if atom.bfactor < 50:
                    self.terminate = True
                    return 0
            return 1
                
        else:
            return 0
    
    #atoms with bfactor higher than threshold
    # def accept_atom(self, atom):
    #     if atom.bfactor > 50:
    #         return 1
    #     else:
    #         return 0

    
        


#print(red_atoms)
io = PDBIO()
io.set_structure(structure)
io.save("data\\red_only.pdb", RedSelect())

# Iterate over atoms and retrieve B-factors
# red_atoms = []
# for model in structure:
#     for chain in model:
#         for residue in chain:
#             for atom in residue:
#                 # Check if the atom has a B-factor value
#                 if atom.is_disordered() or atom.bfactor is None:
#                     continue
#                 # Define a threshold for what we consider "red"
#                 threshold = 50  # Adjust as needed
#                 if atom.bfactor > threshold:
#                     red_atoms.append(atom)

