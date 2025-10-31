import numpy as np
import matplotlib.pyplot as plt

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



def impute_missing_values(X_imputed, strategy="normal"):
    """
    Replace NaN values with column-wise mean, median or normal sampling.
    strategy: "mean", "median" or "normal"
    """
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


def standardize_features(X, num_mask=None, means=None, stds=None):
    """
    Standardizes numerical features (mean 0, std 1).
    Non-numerical (categorical / one-hot) features are left unchanged.

    Args:
        X : np.ndarray
            Data matrix (n_samples, n_features)
        num_mask : np.ndarray of bool
            Boolean mask where True means the feature is numerical.
        means : np.ndarray or None
            Optional predefined means for numerical features (used for val/test)
        stds : np.ndarray or None
            Optional predefined stds for numerical features (used for val/test)

    Returns:
    
        X_std : np.ndarray
            Standardized copy of X.
        means : np.ndarray
            Means used for numerical features.
        stds : np.ndarray
            Standard deviations used for numerical features.
    """
    X_std = X.copy().astype(float)
    if num_mask is None: num_mask = np.ones(X_std.shape[1], dtype=bool)
    idx_num = np.where(num_mask)[0]

    if means is None or stds is None:
        means = np.mean(X_std[:, idx_num], axis=0)
        stds = np.std(X_std[:, idx_num], axis=0)
        stds[stds == 0] = 1.0  # avoid division by zero
    X_std[:, idx_num] = (X_std[:, idx_num] - means) / stds
    return X_std, means, stds



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


# Numerical and categorical features

def detect_features_type(X, max_unique=30):
    """
    Detects categorical and numerical columns in a numeric NumPy array.

    Args:
        X : np.ndarray
            Numeric data matrix (N, d)
        max_unique : int
            Maximum number of unique values to consider the column categorical.

    Returns:
        cat_mask : np.ndarray of bool (d, 1)
            Boolean mask of categorical columns.
        num_mask : np.ndarray of bool (d, 1)
            Boolean mask of numerical columns.
    """
    d = X.shape[1]
    cat_mask = np.zeros(d, dtype=bool)

    for j in range(d):
        col = X[:, j]
        unique_vals = np.unique(col[~np.isnan(col)])
        if len(unique_vals) <= max_unique:
            cat_mask[j] = True

    num_mask = ~cat_mask
    return cat_mask, num_mask

def impute_numerical(X_imp, num_mask, medians=None, means=None, strategy='medians'):
    """
    Imputes NaN values in numerical features with their medians, means, or samples from a normal distribution.
    If predefined medians or means are provided, they are used instead.

    Args:
        X : np.ndarray
            Data matrix (n_samples, n_features)
        num_mask : np.ndarray of bool
            Boolean mask where True means the feature is numerical.
        medians : np.ndarray or None, optional
            Predefined medians for numerical features (used if not None).
        means : np.ndarray or None, optional
            Predefined means for numerical features (used if not None).
        strategy : {'medians', 'means', 'normal'}
            Strategy used for imputation.

    Returns:
        X_imp : np.ndarray
            Copy of X with NaNs in numerical features imputed.
        values : np.ndarray
            Values used (medians or means).
    """
    idx_num = np.where(num_mask)[0]

    if strategy == 'medians':
        if medians is None:
            values = np.nanmedian(X_imp[:, idx_num], axis=0)
        else:
            values = medians

    else:  # means or normal
        if means is None:
            values = np.nanmean(X_imp[:, idx_num], axis=0)
        else:
            values = means
    # Replace NaNs
    for i, j in enumerate(idx_num):
        col = X_imp[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            if strategy == 'normal':
                std = np.nanstd(col)
                col[nan_mask] = np.random.normal(values[i], std, size=nan_mask.sum())
            else:
                col[nan_mask] = values[i]
            X_imp[:, j] = col
    return X_imp, values

def impute_categorical(X, cat_mask, modes=None):
    """
    Imputes NaN values in categorical features with the mode (most frequent value).
    If predefined modes are provided, they are used instead.

    Args:
        X : np.ndarray
            Data matrix (n_samples, n_features)
        cat_mask : np.ndarray of bool
            Boolean mask where True means the feature is categorical.
        modes : np.ndarray or None, optional
            Predefined modes for categorical features (used if not None).

    Returns:
        X_imp : np.ndarray
            Copy of X with NaNs in categorical features imputed.
        modes : np.ndarray
            Mode values used for each categorical feature.
    """
    X_imp = X.copy()
    idx_cat = np.where(cat_mask)[0]

    if modes is None:
        modes = np.zeros(len(idx_cat))
        for i, j in enumerate(idx_cat):
            col = X_imp[:, j]
            col_nonan = col[~np.isnan(col)]
            if col_nonan.size == 0:
                modes[i] = np.nan  # all missing
            else:
                vals, counts = np.unique(col_nonan, return_counts=True)
                modes[i] = vals[np.argmax(counts)]
    else:
        modes = modes.copy()

    for i, j in enumerate(idx_cat):
        col = X_imp[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            col[nan_mask] = modes[i]
            X_imp[:, j] = col

    return X_imp, modes


def one_hot_encode(X, cat_mask, categories=None, drop_first=True):
    """
    One-hot encodes categorical features.

    Args:
        X : np.ndarray
            Data matrix (n_samples, n_features)
        cat_mask : np.ndarray of bool
            Boolean mask where True means the feature is categorical.
        categories : list of np.ndarray or None
            List containing the unique values for each categorical feature.
            If None, categories are computed from X (use this for training).
        drop_first : bool
            If True, drops the first category to avoid collinearity.

    Returns:
    
        X_enc : np.ndarray
            Matrix with numerical features kept and categorical features one-hot encoded.
        categories : list of np.ndarray
            List of category values used for encoding each categorical feature.
    """
    X_enc_parts = []
    n_samples, n_features = X.shape
    idx_cat = np.where(cat_mask)[0]
    idx_num = np.where(~cat_mask)[0]

    # Keep numeric features as is
    if len(idx_num) > 0:
        X_enc_parts.append(X[:, idx_num].astype(float))

    # Encode categorical features
    if categories is None:
        categories = [np.unique(X[:, j]) for j in idx_cat]

    for j, cats in zip(idx_cat, categories):
        if drop_first and len(cats) > 1:
            cats = cats[1:]  # drop first category to avoid dummy trap
        for c in cats:
            col = (X[:, j] == c).astype(float)
            X_enc_parts.append(col.reshape(-1, 1))

    # Concatenate numeric and encoded categorical parts
    X_enc = np.hstack(X_enc_parts).astype(float)
    return X_enc, categories

# Correlation elimination functions´

def feature_target_correlation(X, y, threshold_elimination=0.01):
    """
    Computes Pearson correlation between each feature and the target y
    and returns a boolean mask indicating features with |correlation| above threshold.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_samples, n_features)
    y : np.ndarray
        Target vector (n_samples,)
    threshold_elimination : float
        Threshold below which features are considered uncorrelated

    Returns
    -------
    mask : np.ndarray
        Boolean array of length n_features (1 if |corr| >= threshold, 0 otherwise)
    """
    Xc = X - np.mean(X, axis=0)
    yc = y - np.mean(y)
    num = np.sum(Xc * yc[:, None], axis=0)
    denom = np.sqrt(np.sum(Xc ** 2, axis=0) * np.sum(yc ** 2))
    corr = np.zeros(X.shape[1])
    valid = denom > 0
    corr[valid] = num[valid] / denom[valid]

    # boolean mask based on threshold
    mask = np.abs(corr) >= threshold_elimination
    return mask

def correlation_matrix(X):
    """
    Computes the Pearson correlation matrix between all features.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_samples, n_features)

    Returns
    -------
    corr_mat : np.ndarray
        (n_features, n_features) correlation matrix.
    """
    Xc = X - np.mean(X, axis=0)
    stds = np.std(Xc, axis=0)
    stds[stds == 0] = 1.0
    Xn = Xc / stds
    corr_mat = (Xn.T @ Xn) / (X.shape[0] - 1)
    return corr_mat


def tvalues_logreg_penalized(w, X, y, lam, pos_weight=1.0, neg_weight=1.0):
    n, d = X.shape
    w = w.reshape(-1,1)
    y = y.reshape(-1,1)
    Xb = np.hstack([np.ones((n,1)), X])
    z = Xb @ w
    p = 1 / (1 + np.exp(-z))
    sample_w = np.where(y == 1, pos_weight, neg_weight).reshape(-1,1)
    w_diag = (p * (1 - p) * sample_w).flatten()
    H_diag = np.sum(Xb**2 * w_diag[:, None], axis=0) + lam
    se = 1 / np.sqrt(H_diag)
    t_vals = (w.flatten() / se)
    return ((t_vals[1:] > 2) | (t_vals[1:] < -2))
