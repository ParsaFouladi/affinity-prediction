import torch
import numpy as np
from torch.utils.data import DataLoader
from data_loader import ProteinLigandTest  
from model_logic import CNNModelBasic,DeeperCNNModel,VGG16,ResNet34
from evaluation_statistics import calculate_metrics
import logging
import argparse
import datetime
import pandas as pd

# Load Trained Model
##model_path = "model.pt"  # Replace with your actual model path
def main(args):
    # Initialize logger
    
        
    current_date = datetime.datetime.now().strftime("%d%m%Y")
    logging.basicConfig(filename=f'{args.log_file}_{current_date}.log', level=logging.INFO, 
                        format='%(asctime)s %(levelname)s:%(message)s')

    # Load the model
    #input_shape = (3, 401, 401)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (args.input_channels, args.height, args.width)
    if args.model_type == 'basic':
        model = CNNModelBasic(input_shape).to(device)
    elif args.model_type == 'deeper':
        model = DeeperCNNModel(input_shape).to(device)
    elif args.model_type == 'VGG16':
        model = VGG16(input_shape=input_shape).to(device)
    elif args.model_type == 'ResNet34':
        model = ResNet34(input_shape=input_shape).to(device)
    else:
        raise ValueError("Invalid model type. Choose from 'basic' or 'deeper'.")
    #model = CNNModelBasic(input_shape)  # Instantiate the model first
    state_dict = torch.load(args.model_path)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove the 'module.' prefix
        else:
            new_state_dict[k] = v
    model.load_state_dict(state_dict=new_state_dict)  # Load model state dictionary
    
    model.to(device) 
    model.eval() 

    # Load Test Dataset (replace with your actual path to the test HDF5 file)
    test_dataset = ProteinLigandTest(args.test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)  # No shuffling during testing

    # Make Predictions and Calculate Metrics
    all_preds = []
    all_targets = []
    results_dict = {}
    with torch.no_grad():
        for batch_id,(representations, binding_affinities,group_names) in enumerate(test_loader):
            logging.info(f"Batch {batch_id} Started")
            representations, binding_affinities= representations.to(device), binding_affinities.to(device)
            outputs = model(representations)
            all_preds.extend(outputs.cpu().numpy().flatten())  # Flatten in case of batches
            all_targets.extend(binding_affinities.cpu().numpy())
            results_dict.update({group_name: (output.cpu().numpy().flatten(), binding_affinity.cpu().numpy()) for group_name, output, binding_affinity in zip(group_names, outputs, binding_affinities)})
            logging.info(f"Batch {batch_id} Completed")

    test_metrics = calculate_metrics(np.array(all_targets), np.array(all_preds))
    logging.info(f"Test Metrics: {test_metrics}")
    # Save Predictions and Metrics
    results_df = pd.DataFrame(results_dict).T
    results_df.columns = ['Predicted', 'Actual']
    results_name=args.result_name
    results_df.to_csv(f"results/{results_name}_{current_date}.csv")
    
    

if __name__=="__main__":
    parser=argparse.ArgumentParser(description='Protein-Ligand Binding Affinity Prediction')
    parser.add_argument('--test_data_path', type=str, default='representation_test.h5', help='Path to the HDF5 test data file')
    # get the model path
    parser.add_argument('--model_path', type=str, default='model.pt', help='Path to the trained model file')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--log_file', type=str, default='test', help='File to save the test logs')
    # get the number of input channels
    parser.add_argument('--input_channels', type=int, default=3, help='Number of input channels')
    # get the hight
    parser.add_argument('--height', type=int, default=401, help='Height of the input image')
    # get the width
    parser.add_argument('--width', type=int, default=401, help='Width of the input image')
    #output file of the results
    parser.add_argument('-o','--result_name', type=str, default='results', help='Output file name')
    # model type
    parser.add_argument('-m','--model_type', type=str, default='basic', help='Type of model to use')

    args=parser.parse_args()
    main(args)
