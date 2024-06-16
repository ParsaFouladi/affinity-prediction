from openbabel import openbabel
import sys
import os

def convert_pdbqt_to_pdb(input_pdbqt_path, output_pdb_path):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdbqt", "pdb")
    
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, input_pdbqt_path)  # Read the PDBQT file
    
    obConversion.WriteFile(mol, output_pdb_path)  # Write the PDB file

# Get the input folder and output folder address from the terminal
if len(sys.argv) > 1:
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    # Convert all PDBQT files in the input folder to PDB files in the output folder
    
    file_number = 0
    for file in os.listdir(input_folder):
        if file.endswith(".pdbqt"):
            input_pdbqt_path = os.path.join(input_folder, file)
            output_pdb_path = os.path.join(output_folder, file.replace(".pdbqt", ".pdb"))
            convert_pdbqt_to_pdb(input_pdbqt_path, output_pdb_path)
            file_number += 1
            print(f"Converted {file_number} PDBQT files to PDB format!")
