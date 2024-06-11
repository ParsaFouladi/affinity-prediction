from openbabel import openbabel
import sys
import os

def convert_pdb_to_pdbqt(input_pdb_path, output_pdbqt_path):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "pdbqt")
    
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, input_pdb_path)  # Read the PDB file
    
    # Optional: add charges, if needed
    # Adding Gasteiger charges
    chargeModel = openbabel.OBChargeModel.FindType("gasteiger")
    chargeModel.ComputeCharges(mol)
    
    obConversion.WriteFile(mol, output_pdbqt_path)  # Write the PDBQT file

# Get the input folder and output folder adress from the terminal
if len(sys.argv)>1:
    input_folder=sys.argv[1]
    output_folder=sys.argv[2]
    # Convert all PDB files in the input folder to PDBQT files in the output folder
    
    file_number=0
    for file in os.listdir(input_folder):
        if file.endswith(".pdb"):
            input_pdb_path = os.path.join(input_folder, file)
            output_pdbqt_path = os.path.join(output_folder, file.replace(".pdb", ".pdbqt"))
            convert_pdb_to_pdbqt(input_pdb_path, output_pdbqt_path)
            file_number+=1
            print(f"Converted {file_number} PDB files to PDBQT format!")
