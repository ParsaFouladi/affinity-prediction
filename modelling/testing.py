import torch
import numpy as np
from torch.utils.data import DataLoader
from data_loader import ProteinLigandTest  
from model_logic import CNNModelBasic          
from training import calculate_metrics
import logging
import argparse
import datetime

# Load Trained Model
##model_path = "model.pt"  # Replace with your actual model path
def main(args):
    # Initialize logger
    
        
    current_date = datetime.datetime.now().strftime("%d%m%Y")
    logging.basicConfig(filename=f'{args.log_file}_{current_date}.log', level=logging.INFO, 
                        format='%(asctime)s %(levelname)s:%(message)s')

    # Load the model
    input_shape = (3, 401, 401)
    model = CNNModelBasic(input_shape)  # Instantiate the model first
    model.load_state_dict(torch.load(args.model_path),strict=False)  # Load model state dictionary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) 
    model.eval() 

    # Load Test Dataset (replace with your actual path to the test HDF5 file)
    test_dataset = ProteinLigandTest(args.test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)  # No shuffling during testing

    # Make Predictions and Calculate Metrics
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_id,(representations, binding_affinities) in enumerate(test_loader):
            logging.info(f"Batch {batch_id} Started")
            representations, binding_affinities = representations.to(device), binding_affinities.to(device)
            outputs = model(representations)
            all_preds.extend(outputs.cpu().numpy().flatten())  # Flatten in case of batches
            all_targets.extend(binding_affinities.cpu().numpy())
            logging.info(f"Batch {batch_id} Completed")

    test_metrics = calculate_metrics(np.array(all_targets), np.array(all_preds))
    logging.info(f"Test Metrics: {test_metrics}")
    
    

if __name__=="__main__":
    parser=argparse.ArgumentParser(description='Protein-Ligand Binding Affinity Prediction')
    parser.add_argument('--test_data_path', type=str, default='representation_test.h5', help='Path to the HDF5 test data file')
    # get the model path
    parser.add_argument('--model_path', type=str, default='model.pt', help='Path to the trained model file')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--log_file', type=str, default='test', help='File to save the test logs')

    args=parser.parse_args()
    main(args)
