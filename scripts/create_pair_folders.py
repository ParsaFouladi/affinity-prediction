import argparse
import os



def preprocess_protein(protein_input):
    #Go through all the files end with .pdb
    for file in os.listdir(protein_input):
        if file.endswith(".pdb"):
            #if file name has a space, replace it with underscore
            if " " in file:
                os.rename(os.path.join(protein_input, file), os.path.join(protein_input, file.replace(" ", "_")))

def create_pair_folders(protein_input, ligand_input, output):
    preprocess_protein(protein_input)

    for file in os.listdir(protein_input):
        if file.endswith(".pdb"):
            for ligand in os.listdir(ligand_input):
                if ligand.endswith(".mol2"):
                    #get the name of the protein
                    protein_name = file.split(".")[0]
                    #get the name of the ligand
                    # ligand_name = ligand.split(".")[0]
                    ligand_name = 'cortisol'
                    if protein_name in ligand:
                        #create a folder for the complex
                        os.makedirs(os.path.join(output, f"{protein_name}_{ligand_name}"))
                        #copy the protein file to the complex folder
                        os.system(f"copy {os.path.join(protein_input, file)} {os.path.join(output, f'{protein_name}_{ligand_name}', file)}")
                        #copy the ligand file to the complex folder
                        os.system(f"copy {os.path.join(ligand_input, ligand)} {os.path.join(output, f'{protein_name}_{ligand_name}', ligand)}")

                        
            # #get the name of the protein
            # protein_name = file.split(".")[0]
            # #get the name of the ligand
            # ligand_name = f"cortisol"
            # #create a folder for the complex
            # os.makedirs(os.path.join(output, f"{protein_name}_{ligand_name}"))
            # #copy the protein file to the complex folder
            # os.system(f"cp {os.path.join(protein_input, file)} {os.path.join(output, f'{protein_name}_{ligand_name}', file)}")
            # #copy the ligand file to the complex folder
            # os.system(f"cp {os.path.join(ligand_input, f'{ligand_name}.mol2')} {os.path.join(output, f'{protein_name}_{ligand_name}', f'{ligand_name}.mol2')}")

            # number += 1




def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Gather protein and ligand pair together.")
    parser.add_argument("-p", "--protein_input", required=True, help="Input folder for protein files")
    parser.add_argument("-l", "--ligand_input", required=True, help="Input folder for ligand files")
    parser.add_argument("-o", "--output", required=True, help="Output folder for the complexes.")
    # #specify number of ligands
    # parser.add_argument("-n", "--number", required=True, help="Number of ligands")
    
    # Parse the arguments
    args = parser.parse_args()
    
    output_folder = args.output
    
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    create_pair_folders(args.protein_input, args.ligand_input, args.output)

if __name__ == "__main__":
    main()