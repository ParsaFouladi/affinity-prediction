from data_loader import ProteinLigandTrain, ProteinLigandTest
from model_logic import CNNModelBasic,DeeperCNNModel,VGG16
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error,mean_absolute_error
import os
import logging
import datetime
from collections import Counter





def calculate_metrics(y_true, y_pred):
    """Calculates regression metrics for binding affinity prediction.

    Args:
        y_true (np.ndarray): True binding affinity values.
        y_pred (np.ndarray): Predicted binding affinity values.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """

    # convert to numpy arrays as float32
    y_true = np.array(y_true, dtype=np.float32).ravel()
    y_pred = np.array(y_pred, dtype=np.float32).ravel()

    
    mse = mean_squared_error(y_true, y_pred)
    mae=mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)
    
    residuals = y_true - y_pred
    sd_residuals = np.std(residuals)

    return {
        "RMSE": rmse,
        "Pearson Correlation": pearson_corr,
        "Spearman Correlation": spearman_corr,
        "Mean Squared Error": mse,
        "Mean Absolute Error":mae,
        "Standard Deviation of Residuals": sd_residuals
    }
# Define the target normalization function
def calculate_target_stats(dataset):
    """Calculate the mean and standard deviation of the target variable."""
    targets = []
    for _, target in dataset:
        targets.append(target.item())
    targets = np.array(targets)
    return np.mean(targets), np.std(targets)
def normalize_target(target, mean, std):
    return (target - mean) / std

# Define the denormalization function (to be used during prediction)
def denormalize_target(normalized_target, mean, std):
    return normalized_target * std + mean
    #Initialize logger
def main(args):
    
    # put the time in the log file name to avoid overwriting in DDMMYYYY format
    # Get current date in DDMMYYYY format
    current_date = datetime.datetime.now().strftime("%d%m%Y")

    logging.basicConfig(filename=f'{args.log_file}_{current_date}.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s')
    # pre_data = ProteinLigandTrain(args.data_path)
    # mean, std = calculate_target_stats(pre_data)
    # logging.info(f"Mean: {mean}, Std: {std}")
    # pre_data.close()
    # # Data Loading
    # # add normalize the target values
    # target_transform = lambda x: normalize_target(x, mean, std)
    dataset = ProteinLigandTrain(args.data_path, target_transform=None)
    train_sampler, val_sampler = dataset.get_train_val_split(args.val_split, args.seed)
    logging.info(f"Training samples: {len(train_sampler)}")

    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler)
    logging.info(f"Validation samples: {len(val_sampler)}")

    # Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    criterion = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=True,min_lr=5e-6)

    # TensorBoard for Logging
    writer = SummaryWriter(log_dir=args.log_dir)

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
            logging.info(f"Validation {metric_name}: {metric_value}")
            writer.add_scalar(f"Metrics/Validation/{metric_name}", metric_value, epoch)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        logging.info(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Step the scheduler
        scheduler.step(val_loss)

        for param_group in optimizer.param_groups:
            logging.info(f"Learning rate at epoch {epoch+1} is: {param_group['lr']}")
            writer.add_scalar('Learning Rate', param_group['lr'], epoch+1)

    writer.close()

    model_name = args.output_name
    # Saving the model
    torch.save(model.state_dict(), os.path.join(args.save_dir, f'{model_name}_{current_date}.pt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Protein-Ligand Binding Affinity Prediction')
    parser.add_argument('--data_path', type=str, default='representations.h5', help='Path to the HDF5 data file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data splitting')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for saving TensorBoard logs')
    parser.add_argument('--save_dir', type=str, default='./models', help='Directory for saving the model')
    parser.add_argument('--log_file', type=str, default='modelling_log', help='Log file name')
    # get the number of input channels
    parser.add_argument('-c','--input_channels', type=int, default=3, help='Number of input channels')
    # get the hight
    parser.add_argument('--height', type=int, default=401, help='Height of the input image')
    # get the width
    parser.add_argument('-w','--width', type=int, default=401, help='Width of the input image')
    #output name
    parser.add_argument('-o','--output_name', type=str, default='model', help='Name of the output model')
    # model type
    parser.add_argument('-m','--model_type', type=str, default='basic', help='Type of model to use')

    args = parser.parse_args()
    main(args)