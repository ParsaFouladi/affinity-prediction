import openbabel
import os
import sys

def convert_sdf_to_pdbqt(input_sdf, output_pdbqt, flexible_residues=False):
    # Create an Open Babel conversion object
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("sdf", "pdbqt")

    # Read the input file
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, input_sdf)

    # Add hydrogens to the molecule
    mol.AddHydrogens()

    # Handle the flexible residues option
    if flexible_residues:
        obConversion.AddOption("xr", openbabel.OBConversion.OUTOPTIONS)

    # Write the molecule to a PDBQT file
    obConversion.WriteFile(mol, output_pdbqt)

# Get the input folder and output folder address from the terminal
if len(sys.argv) > 1:
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    flexible_residues = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Convert all SDF files in the input folder to PDBQT files in the output folder
    file_number = 0
    for file in os.listdir(input_folder):
        if file.endswith(".sdf"):
            input_sdf_path = os.path.join(input_folder, file)
            output_pdbqt_path = os.path.join(output_folder, file.replace(".sdf", ".pdbqt"))
            convert_sdf_to_pdbqt(input_sdf_path, output_pdbqt_path, flexible_residues)
            file_number += 1
            print(f"Converted {file_number} SDF files to PDBQT format!")
else:
    print("Usage: python convert_sdf_to_pdbqt.py <input_folder> <output_folder> [flexible_residues]")
    print("Example: python convert_sdf_to_pdbqt.py ./input ./output true")
