import numpy as np

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


def impute_missing_values(X, strategy="mean"):
    """
    Replace NaN values with column-wise mean or median.
    strategy: "mean" or "median"
    """
    X_imputed = X.copy()
    if strategy == "mean":
        values = np.nanmean(X_imputed, axis=0)
    elif strategy == "median":
        values = np.nanmedian(X_imputed, axis=0)
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


def clean_data(X, y=None, feature_nan_threshold=0.3, impute_strategy="mean", drop_nan_rows=False, standardize=True):
    """
    Full cleaning pipeline:
      1. Remove features with too many NaNs
      2. Impute remaining NaNs
      3. Optionally remove rows with NaNs
      4. Standardize features
    """
    # Step 1: remove features with too many NaNs
    X, feature_mask = remove_nan_features(X, threshold=feature_nan_threshold)

    # Step 2: impute missing values
    X, impute_values = impute_missing_values(X, strategy=impute_strategy)

    # Step 3: optionally remove remaining rows with NaNs
    row_mask = None
    if drop_nan_rows:
        X, y, row_mask = remove_nan_rows(X, y) if y is not None else remove_nan_rows(X)

    # Step 4: standardize
    if standardize:
        X, means, stds = standardize_features(X)
    else:
        means, stds = None, None

    return {
        "X_clean": X,
        "y_clean": y,
        "feature_mask": feature_mask,
        "row_mask": row_mask,
        "impute_values": impute_values,
        "means": means,
        "stds": stds
    }

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

