import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm

def sigmoid(z):

    # Clip z to avoid overflow in exp()
    z = np.clip(z, -700, 700)  # exp(709) is close to float64 max (~1e308)

    return 1.0 / (1.0 + np.exp(-z))


def remove_nan_features(X, threshold=0.3):
    """
    Remove columns (features) with more than `threshold` proportion of NaN values.
    """
    nan_per_feature = np.sum(np.isnan(X), axis=0)
    keep_mask = nan_per_feature < threshold * X.shape[0]
    X_clean = X[:, keep_mask]
    return X_clean, keep_mask


def remove_nan_rows(X, y=None):
    """
    Remove rows (samples) that contain any NaN values.
    """
    row_mask = ~np.any(np.isnan(X), axis=1)
    X_clean = X[row_mask]
    if y is not None:
        y_clean = y[row_mask]
        return X_clean, y_clean, row_mask
    return X_clean


def impute_missing_values(X, strategy="normal"):
    """
    Replace NaN values with column-wise mean, median or normal sampling.
    strategy: "mean", "median" or "normal"
    """
    X_imputed = X.copy()
    n_features = X_imputed.shape[1]
    values = []
    if strategy == "mean":
        values = np.nanmean(X_imputed, axis=0)
    elif strategy == "median":
        values = np.nanmedian(X_imputed, axis=0)
    elif strategy == "normal":
        means = np.nanmean(X_imputed, axis=0)
        stds = np.nanstd(X_imputed, axis=0)

        for j in range(n_features):
            nan_mask = np.isnan(X_imputed[:, j])
            n_missing = np.sum(nan_mask)
            if n_missing > 0:
                # Draw samples from N(mean_j, std_j)
                samples = np.random.normal(loc=means[j], scale=stds[j], size=n_missing)
                X_imputed[nan_mask, j] = samples
    else:
        raise ValueError("Invalid strategy: choose 'mean' or 'median'.")

    inds = np.where(np.isnan(X_imputed))
    X_imputed[inds] = np.take(values, inds[1])
    return X_imputed


def standardize_features(X):
    """
    Standardize features to have mean 0 and std 1.
    """
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    stds[stds == 0] = 1.0  # avoid division by zero
    X_std = (X - means) / stds
    return X_std, means, stds



def select_random_features(X, n):
    """
    Randomly select n features (columns) from X.
    """
    idx = np.random.choice(X.shape[1], n, replace=False)
    return X[:, idx]

def balance_data(X, y, method="undersample", random_state=None):
    """
    Balances a binary dataset using only NumPy.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Binary target vector (0/1).
    method : str, optional, default="undersample"
        - "undersample": randomly drop majority samples
        - "oversample": randomly duplicate minority samples
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    X_bal : np.ndarray
        Balanced feature matrix.
    y_bal : np.ndarray
        Balanced target vector.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Separate indices by class
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    
    n_0, n_1 = len(idx_0), len(idx_1)

    if method == "undersample":
        # Downsample majority class
        if n_0 > n_1:
            idx_0_sampled = np.random.choice(idx_0, size=n_1, replace=False)
            balanced_idx = np.concatenate([idx_0_sampled, idx_1])
        else:
            idx_1_sampled = np.random.choice(idx_1, size=n_0, replace=False)
            balanced_idx = np.concatenate([idx_0, idx_1_sampled])
    
    elif method == "oversample":
        # Upsample minority class
        if n_0 > n_1:
            idx_1_sampled = np.random.choice(idx_1, size=n_0, replace=True)
            balanced_idx = np.concatenate([idx_0, idx_1_sampled])
        else:
            idx_0_sampled = np.random.choice(idx_0, size=n_1, replace=True)
            balanced_idx = np.concatenate([idx_0_sampled, idx_1])
    
    else:
        raise ValueError("method must be 'undersample' or 'oversample'")
    
    # Shuffle combined indices
    np.random.shuffle(balanced_idx)
    
    # Return balanced data
    X_bal = X[balanced_idx]
    y_bal = y[balanced_idx]
    
    return X_bal, y_bal


def clean_data(X_train, X_val, y_train, y_val, random_state=None):
        
        """"
        Args:
            X_train (np.ndarray): shape (n_samples, n_features) with Nan values and unbalanced data
            all the features will be used in this model (otherwise delete befor calling the function)
            X_val (np.ndarray): shape (m_samples, n_features) with Nan values
            y_train (np.ndarray): shape (n_samples,) with binary labels (0/1)
            y_val (np.ndarray): shape (m_samples,) with binary labels (0/1)
            random_state (int or None): For reproducible shuffling.
        Returns:
            X_train (np.ndarray): Cleaned and balanced training data.
            X_val (np.ndarray): Cleaned validation data.
            y_train (np.ndarray): Balanced training labels.
            y_val (np.ndarray): Validation labels.
        """
        
        if np.any(y_train == -1): y_train = (y_train + 1) / 2  #change labels if needed
        if np.any(y_val == -1): y_val = (y_val + 1) / 2
        X_train, X_val = impute_missing_values(X_train, strategy="normal"), impute_missing_values(X_val, strategy="normal")
        X_train, X_means, X_stds = standardize_features(X_train)
        X_val = (X_val - X_means) / X_stds # we normalize with the training parameters, not de validation ones
        X_train, y_train = balance_data(X_train, y_train, method="undersample", random_state=random_state)

        return X_train, X_val, y_train, y_val



# Features selection




