import argparse
import os


# The purpose of this script is to fix the names of the ligands and proteins in the complex folders.

# This function will make go through the complex folders and in each folder, it will rename the protein (end with .pdb) to {folder_name}_protein.pdb 
# and ligand (end with .mol2) files to {folder_name}_ligand.mol2.
def fix_names(input_folder):
    for folder in os.listdir(input_folder):
        if os.path.isdir(os.path.join(input_folder, folder)):
            for file in os.listdir(os.path.join(input_folder, folder)):
                if file.endswith(".pdb"):
                    os.rename(os.path.join(input_folder, folder, file), os.path.join(input_folder, folder, f"{folder}_protein.pdb"))
                elif file.endswith(".mol2"):
                    os.rename(os.path.join(input_folder, folder, file), os.path.join(input_folder, folder, f"{folder}_ligand.mol2"))

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Fix the names of the protein and ligand files in the complex folders.")
    parser.add_argument("-i", "--input_folder", required=True, help="Input folder containing the complex folders.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the fix_names function
    fix_names(args.input_folder)

if __name__ == "__main__":
    main()