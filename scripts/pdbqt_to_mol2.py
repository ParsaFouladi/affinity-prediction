import argparse
from openbabel import openbabel
import os

def convert_pdbqt_to_mol2(input_pdbqt_path, output_mol2_path):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdbqt", "mol2")
    
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, input_pdbqt_path)  # Read the PDBQT file
    
    obConversion.WriteFile(mol, output_mol2_path)  # Write the MOL2 file

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert PDBQT files to MOL2 format.")
    parser.add_argument("-i", "--input", required=True, help="Input folder containing PDBQT files.")
    parser.add_argument("-o", "--output", required=True, help="Output folder for MOL2 files.")
    
    # Parse the arguments
    args = parser.parse_args()
    input_folder = args.input
    output_folder = args.output
    
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Convert all PDBQT files in the input folder to MOL2 files in the output folder
    file_number = 0
    for file in os.listdir(input_folder):
        if file.endswith(".pdbqt"):
            input_pdbqt_path = os.path.join(input_folder, file)
            output_mol2_path = os.path.join(output_folder, file.replace(".pdbqt", ".mol2"))
            convert_pdbqt_to_mol2(input_pdbqt_path, output_mol2_path)
            file_number += 1
            print(f"Converted {file_number} PDBQT files to MOL2 format!")

if __name__ == "__main__":
    main()