from data_loader import ProteinLigandTrain, ProteinLigandTest
from model_logic import CNNModelBasic,DeeperCNNModel,VGG16
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from evaluation_statistics import calculate_metrics
import os
import logging
import datetime
from collections import Counter
from sklearn.model_selection import KFold

def main(args):
    # put the time in the log file name to avoid overwriting in DDMMYYYY format
    # Get current date in DDMMYYYY format
    current_date = datetime.datetime.now().strftime("%d%m%Y")

    logging.basicConfig(filename=f'{args.log_file}_{current_date}.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s')
    
    # Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # logging.info(f"Using {device} for training")
    
    # #input_shape = (3, 401, 401)
    # input_shape = (args.input_channels, args.height, args.width)
    # if args.model_type == 'basic':
    #     model = CNNModelBasic(input_shape).to(device)
    # elif args.model_type == 'deeper':
    #     model = DeeperCNNModel(input_shape).to(device)
    # elif args.model_type == 'VGG16':
    #     model = VGG16(input_shape=input_shape).to(device)
    # #model = CNNModelBasic(input_shape).to(device)
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     logging.info(f"Using {torch.cuda.device_count()} GPUs!")
    #     model = nn.DataParallel(model)  # Use DataParallel for multi-GPU training
    
    # state_dict = torch.load(args.model_path)
    # new_state_dict = {}
    # for k, v in state_dict.items():
    #     if k.startswith('module.'):
    #         new_state_dict[k[7:]] = v  # Remove the 'module.' prefix
    #     else:
    #         new_state_dict[k] = v
    # model.load_state_dict(state_dict=new_state_dict)  # Load model state dictionary
    # model.train()

    # criterion = nn.MSELoss()  
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=16, verbose=True,min_lr=4e-5)

    # # TensorBoard for Logging
    # writer = SummaryWriter(log_dir=args.log_dir)

    train_dataset = ProteinLigandTrain(args.data_path)
    # Initialize the k-fold cross validation
    kf = KFold(n_splits=args.kfolds, shuffle=True)

    for fold, (train_index, val_index) in enumerate(kf.split(train_dataset)):
        logging.info(f"Using {device} for training")
    
        #input_shape = (3, 401, 401)
        input_shape = (args.input_channels, args.height, args.width)
        if args.model_type == 'basic':
            model = CNNModelBasic(input_shape).to(device)
        elif args.model_type == 'deeper':
            model = DeeperCNNModel(input_shape).to(device)
        elif args.model_type == 'VGG16':
            model = VGG16(input_shape=input_shape).to(device)
        #model = CNNModelBasic(input_shape).to(device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            logging.info(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)  # Use DataParallel for multi-GPU training
        
        state_dict = torch.load(args.model_path)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove the 'module.' prefix
            else:
                new_state_dict[k] = v
        model.load_state_dict(state_dict=new_state_dict)  # Load model state dictionary
        model.train()
        criterion = nn.MSELoss()  
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=16, verbose=True,min_lr=4e-5)

        # TensorBoard for Logging
        writer = SummaryWriter(log_dir=args.log_dir)
        # restart the model for each fold


        logging.info(f"Fold {fold + 1} Started")
        # Create a dictionary for keeping track of the metrics
        results_dict = {}

        train_loader=DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_index)
        val_loader=DataLoader(train_dataset, batch_size=args.batch_size, sampler=val_index)

        # Training Loop
        for epoch in range(args.epochs):
            logging.info(f"Epoch {epoch + 1}/{args.epochs} Started")
            model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []

            for batch_idx, (representations, binding_affinities) in enumerate(train_loader):
                logging.info(f"Batch {batch_idx + 1}/{len(train_loader)} of size {args.batch_size} of epoch {epoch + 1}/{args.epochs}")

                representations, binding_affinities = representations.to(device), binding_affinities.to(device)

                print(representations.shape)

                optimizer.zero_grad()
                outputs = model(representations)
                # logging.info(f"Outputs: {outputs}")
                # logging.info(f"Targets: {binding_affinities}")
                loss = criterion(outputs, binding_affinities)
                # Weighted MSE Loss
                
                train_loss += loss.item()  # Summing up training loss for the epoch
                loss.backward()
                optimizer.step()

                for out in outputs.cpu().detach().numpy():
                    train_preds.append(out[0])
                    #all_preds.extend(outputs.cpu().numpy())
                train_targets.extend(binding_affinities.cpu().detach().numpy())

                logging.info(f"Batch {batch_idx + 1}/{len(train_loader)} of size {args.batch_size} of epoch {epoch + 1}/{args.epochs} Finished" + " Loss: " + str(loss.item()))
                
            train_loss /= len(train_loader)
            writer.add_scalar("Loss/train", train_loss, epoch)
            logging.info(f"Epoch {epoch + 1} training loss: {train_loss}")

            train_metrics = calculate_metrics(np.array(train_targets), np.array(train_preds))
            # Log training metrics to TensorBoard
            for metric_name, metric_value in train_metrics.items():
                logging.info(f"Training {metric_name}: {metric_value}")
                writer.add_scalar(f"Metrics/Training/{metric_name}", metric_value, epoch)
                
            # Validation
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for representations, binding_affinities in val_loader:
                    representations, binding_affinities = representations.to(device), binding_affinities.to(device)
                    outputs = model(representations)
                    loss = criterion(outputs, binding_affinities)
                    

                    val_loss += loss.item()
                    
                    for out in outputs.cpu().numpy():
                        all_preds.append(out[0])
                    
                    #     logging.info(f"Predictions: {out[0]}")
                    # logging.info(f"Targets: {binding_affinities.cpu().numpy()}")

                    #all_preds.extend(outputs.cpu().numpy())
                    all_targets.extend(binding_affinities.cpu().numpy().ravel())
            
            # print all the predictions and targets
            logging.info(f"Predictions: {all_preds}")
            logging.info(f"Targets: {all_targets}")

            val_loss /= len(val_loader)
            writer.add_scalar("Loss/val", val_loss, epoch)
            logging.info(f"Epoch {epoch + 1} validation loss: {val_loss}")

            val_metrics = calculate_metrics(np.array(all_targets), np.array(all_preds))
            # Log validation metrics to TensorBoard
            for metric_name, metric_value in val_metrics.items():
                # Get the metrics in the last epoch
                if epoch == args.epochs - 1:
                    # Increase each metric by the value in the dictionary
                    results_dict[metric_name] = results_dict.get(metric_name, 0) + metric_value
                
                logging.info(f"Validation {metric_name}: {metric_value}")
                writer.add_scalar(f"Metrics/Validation/{metric_name}", metric_value, epoch)

            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            logging.info(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            # Step the scheduler
            #scheduler.step(val_loss)

            for param_group in optimizer.param_groups:
                logging.info(f"Learning rate at epoch {epoch+1} is: {param_group['lr']}")
                writer.add_scalar('Learning Rate', param_group['lr'], epoch+1)

        logging.info(f"Fold {fold + 1} Completed")
        
       
        writer.close()

    # Calculate the average metrics for all the folds
    for metric_name in results_dict:
        results_dict[metric_name] /= args.kfolds
    logging.info(f"Average Metrics: {results_dict}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Protein-Ligand Binding Affinity Prediction Transfer Learning')
    parser.add_argument('--data_path', type=str, default='representations.h5', help='Path to the HDF5 data file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--kfolds', type=float, default=3, help='Number of folds for cross validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data splitting')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for saving TensorBoard logs')
    parser.add_argument('--save_dir', type=str, default='./models', help='Directory for saving the model')
    parser.add_argument('--log_file', type=str, default='modelling_transfer_learning_log', help='Log file name')
    # get the number of input channels
    parser.add_argument('-c','--input_channels', type=int, default=3, help='Number of input channels')
    # get the hight
    parser.add_argument('--height', type=int, default=301, help='Height of the input image')
    # get the width
    parser.add_argument('-w','--width', type=int, default=301, help='Width of the input image')
    #output name
    parser.add_argument('-o','--output_name', type=str, default='model', help='Name of the output model')
    # model type
    parser.add_argument('-m','--model_type', type=str, default='basic', help='Type of model to use')
    # get the model path
    parser.add_argument('--model_path', type=str, default='model.pt', help='Path to the trained model file')

    args = parser.parse_args()
    main(args)