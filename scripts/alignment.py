from pymol import cmd
import sys
from os import listdir
from os.path import isfile, join

def align_proteins(proteins,reference):
    """
    Aligns the proteins to the reference protein
    """
    # Load the reference protein
    cmd.load(reference, "reference")
    # Align the proteins to the reference protein
    for protein in proteins:
        try:
            cmd.load(protein, "protein")
            cmd.super("protein", "reference")

            protein_name = protein.split("\\")[-1]
            # Save the aligned protein to a new PDB file
            cmd.save("alignment_output/aligned_{}".format(protein_name), "protein")
            # Delete the protein object
            cmd.delete("protein")
        except:
            print("An error occurred while aligning protein:", protein)
            # Delete the reference object
            cmd.delete("reference")
            cmd.delete("protein")

if len(sys.argv) > 1:
    # The first command-line argument (index 0) is the script name
    # So the actual argument provided by the user is at index 1
    user_input = sys.argv[1]
    #print("You provided input:", user_input)
    mypath = user_input
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    #print(onlyfiles)

    # Create a list to store the paths of the proteins
    proteins = []
    for file in onlyfiles:
        proteins.append(mypath+'\\'+file)
    try:
    # Align the proteins to the reference protein
        align_proteins(proteins, "filter_output_original\AF-P49843-F1-model_v4.pdb")
    except:
        print("An error occurred during protein alignment.")
        
