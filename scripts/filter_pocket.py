import sys 
sys.path.append('..')
from helpers.binding_pocket import BindingPocket


#create a binding pocket object
binding_pocket=BindingPocket(file_path="data\AF-A0A087XJA9-F1-model_v4.pdb",output_path="data\\test_class.pdb",pocket="KAIEP",b_factor_threshold=50,num_atoms_before=7)
#find the binding pocket
binding_pocket.find_binding_pocket()