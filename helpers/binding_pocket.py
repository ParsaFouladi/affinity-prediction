from Bio.PDB import *

class BindingPocket:
    def __init__(self,file_path,output_path,pocket,b_factor_threshold=50,num_atoms_before=0):
        #self.protein = protein
        self.pocket = pocket
        self.num_atoms_before=0
        self.parser = PDBParser()
        self.structure = self.parser.get_structure("protein", file_path)
        self.ppb = PPBuilder()
        pep=self.ppb.build_peptides(self.structure)
        self.seq = pep[0].get_sequence()
        self.b_factor_threshold=b_factor_threshold
        self.output_path=output_path
        self.file_path=file_path
        self.num_atoms_before=num_atoms_before
                #self.ligand = ligand

    def get_pocket(self):
        return self.pocket

    def get_protein(self):
        return self.protein

    # def get_ligand(self):
    #     return self.ligand
    
    #Set the protein
    def set_protein(self, protein):
        self.protein=protein

    #Set the pocket
    def set_pocket(self, pocket,num_atoms_before=0):
        self.pocket=pocket
        self.num_atoms_before=num_atoms_before
    
    def find_binding_pocket(self):
        try:
            target_index=self.seq.find("{}".format(self.pocket))
            if target_index==-1:
                print("Pocket not found!")
                return 0

            starting_index=target_index-self.num_atoms_before

            b_factor_threshold=self.b_factor_threshold
            # Define a custom selection class
            class FilterTarget(Select):
                def __init__(self):
                    self.terminate = False

                # Accept the residues with index of higher than starting index and atoms with bfactor higher than threshold
                def accept_residue(self, residue):
                    
                    if residue.get_id()[1] > starting_index and not self.terminate:
                        #get all the atoms in the residue
                        atoms = residue.get_list()
                        for atom in atoms:
                            if atom.bfactor < b_factor_threshold:
                                self.terminate = True
                                return 0
                        return 1
                            
                    else:
                        return 0
            # Save the atoms in the selection to a new PDB file
            io = PDBIO()
            io.set_structure(self.structure)
            io.save(self.output_path, FilterTarget())
            return 1
        
        except Exception as e:
            #print the error
            print(e)
            return 0
    