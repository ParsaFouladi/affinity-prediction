from data_loader import ProteinLigandDataset
from model_logic import CNNModelBasic
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
import os
import logging

#Initialize logger
logging.basicConfig(filename='modelling_log.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s')


def calculate_metrics(y_true, y_pred):
    """Calculates regression metrics for binding affinity prediction.

    Args:
        y_true (np.ndarray): True binding affinity values.
        y_pred (np.ndarray): Predicted binding affinity values.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)

    return {
        "RMSE": rmse,
        "Pearson Correlation": pearson_corr,
        "Spearman Correlation": spearman_corr
    }
def main(args):
    # Data Loading
    dataset = ProteinLigandDataset(args.data_path)
    train_sampler, val_sampler = dataset.get_train_val_split(args.val_split, args.seed)
    logging.info(f"Training samples: {len(train_sampler)}")

    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler)
    logging.info(f"Validation samples: {len(val_sampler)}")

    # Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using {device} for training")

    input_shape = (3, 401, 401)
    model = CNNModelBasic(input_shape).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)  # Use DataParallel for multi-GPU training
    criterion = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # TensorBoard for Logging
    writer = SummaryWriter(log_dir=args.log_dir)

    # Training Loop
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch + 1}/{args.epochs} Started")
        model.train()
        train_loss = 0.0

        for batch_idx, (representations, binding_affinities) in enumerate(train_loader):
            logging.info(f"Batch {batch_idx + 1}/{len(train_loader)} of size {args.batch_size} of epoch {epoch + 1}/{args.epochs}")

            representations, binding_affinities = representations.to(device), binding_affinities.to(device)

            print(representations.shape)

            optimizer.zero_grad()
            outputs = model(representations)
            loss = criterion(outputs, binding_affinities)
            train_loss += loss.item()  # Summing up training loss for the epoch
            loss.backward()
            optimizer.step()

            logging.info(f"Batch {batch_idx + 1}/{len(train_loader)} of size {args.batch_size} of epoch {epoch + 1}/{args.epochs} Finished" + " Loss: " + str(loss.item()))
            
        train_loss /= len(train_loader)
        writer.add_scalar("Loss/train", train_loss, epoch)
            
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
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(binding_affinities.cpu().numpy())

        val_loss /= len(val_loader)
        writer.add_scalar("Loss/val", val_loss, epoch)

        val_metrics = calculate_metrics(np.array(all_targets), np.array(all_preds))
        # Log validation metrics to TensorBoard
        for metric_name, metric_value in val_metrics.items():
            writer.add_scalar(f"Metrics/{metric_name}", metric_value, epoch)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        logging.info(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    writer.close()

    # Saving the model
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'model.pt'))

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

    args = parser.parse_args()
    main(args)