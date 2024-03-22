import sys 
sys.path.append('..')
import os
from helpers.binding_pocket import BindingPocket
from os import listdir
from os.path import isfile, join

# Check if the user provided an argument
if len(sys.argv) > 1:
    # The first command-line argument (index 0) is the script name
    # So the actual argument provided by the user is at index 1
    user_input = sys.argv[1]
    #print("You provided input:", user_input)
    mypath = user_input
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    #print(onlyfiles)

    for file in onlyfiles:
        #create a binding pocket object
        binding_pocket=BindingPocket(file_path=mypath+'\\'+file,output_path="filter_output\\"+file,pocket="KAIEP",b_factor_threshold=50,num_atoms_before=7)
        #find the binding pocket
        binding_pocket.find_binding_pocket()

    #create a binding pocket object
    # binding_pocket=BindingPocket(file_path="data\AF-A0A087XJA9-F1-model_v4.pdb",output_path="data\\test_class.pdb",pocket="KAIEP",b_factor_threshold=50,num_atoms_before=7)
    # #find the binding pocket
    # binding_pocket.find_binding_pocket()
else:
    print("No input provided.")

