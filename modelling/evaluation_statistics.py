import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error,mean_absolute_error

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