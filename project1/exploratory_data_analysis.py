import numpy as np
import matplotlib.pyplot as plt


#NaN analysis
def nan_count(X, plot=False):
    """
    Counts the number and percentage of NaN values per column.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix (n_samples, n_features)
    plot : bool, optional
        If True, plots a histogram of NaN counts per column.

    Returns
    -------
    nan_number : np.ndarray
        Number of NaN values per column.
    nan_pctg : np.ndarray
        Percentage of NaN values per column.
    """
    n = X.shape[0]
    nan_mask = np.isnan(X)
    nan_number = np.sum(nan_mask, axis=0)
    nan_pctg = nan_number / n * 100

    if plot:
        plt.figure(figsize=(6,4))
        plt.hist(nan_number, bins=100, edgecolor='black')
        plt.xlabel('Number of NaN values per column')
        plt.ylabel('Count of features')
        plt.title('NaN distribution across features')
        plt.show()

    return nan_number, nan_pctg

#Detecting categorical and numerical features, to treat them differently

def detect_categorical_features(X, max_unique=30):
    """
    Detects categorical columns in a numeric NumPy array.

    Parameters
    ----------
    X : np.ndarray
        Numeric data matrix (n_samples, n_features)
    max_unique : int
        Maximum number of unique values to consider the column categorical.

    Returns
    -------
    cat_mask : np.ndarray of bool
        Boolean mask of categorical columns.
    num_mask : np.ndarray of bool
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


#Calculation of basic stats of the data

def summarize_basic(X, cat_mask):
    """
    Computes basic statistics for each column, including mode for categorical ones.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_samples, n_features)
    cat_mask : np.ndarray of bool
        Boolean mask where True means the column is categorical.

    Returns
    -------
    stats : dict of np.ndarray
        Dictionary containing mean, std, min, max, median, and mode.
    """
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)
    mins = np.nanmin(X, axis=0)
    maxs = np.nanmax(X, axis=0)
    medians = np.nanmedian(X, axis=0)

    # mode only for categorical columns
    d = X.shape[1]
    modes = np.full(d, np.nan)

    for j in range(d):
        if cat_mask[j]:
            col = X[:, j]
            col = col[~np.isnan(col)]
            if len(col) == 0:
                continue
            values, counts = np.unique(col, return_counts=True)
            modes[j] = values[np.argmax(counts)]

    stats = {
        "mean": means,
        "std": stds,
        "min": mins,
        "max": maxs,
        "median": medians,
        "mode": modes
    }

    return stats

def show_feature_stats(stats, j):
    print(f"Feature {j}")
    print(f"  Mean:   {stats['mean'][j]:.4f}")
    print(f"  Std:    {stats['std'][j]:.4f}")
    print(f"  Min:    {stats['min'][j]:.4f}")
    print(f"  Max:    {stats['max'][j]:.4f}")
    print(f"  Median: {stats['median'][j]:.4f}")
    print(f"  Mode:   {stats['mode'][j]}")


#Plot features

def plot_feature_distribution(X, cat_mask, feature_index, bins=30):
    """
    Plots the distribution of a feature.

    Parameters
    ----------
    X : np.ndarray
        Numeric data matrix (n_samples, n_features)
    cat_mask : np.ndarray of bool
        Boolean mask indicating categorical features
    feature_index : int
        Index of the feature to plot
    bins : int
        Number of bins for histograms (numerical features only)
    """
    col = X[:, feature_index]
    col = col[~np.isnan(col)]

    if cat_mask[feature_index]:
        # Bar plot for categorical feature
        values, counts = np.unique(col, return_counts=True)
        plt.figure(figsize=(6, 4))
        plt.bar(values, counts, edgecolor='black')
        plt.xlabel(f'Feature {feature_index} (categorical)')
        plt.ylabel('Count')
        plt.title(f'Distribution of feature {feature_index}')
        plt.show()
    else:
        # Histogram for numerical feature
        plt.figure(figsize=(6, 4))
        plt.hist(col, bins=bins, edgecolor='black')
        plt.xlabel(f'Feature {feature_index} (numerical)')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of feature {feature_index}')
        plt.show()
        
#Outlier detection

def detect_outliers(X, num_mask, threshold=1.5, plot=False):
    """
    Detects outliers in numerical features using the IQR method.

    Parameters
    ----------
    X : np.ndarray
        Numeric data matrix (n_samples, n_features)
    num_mask : np.ndarray of bool
        Boolean mask indicating numerical features
    threshold : float, optional
        Multiplier for the IQR (default = 1.5)
    plot : bool, optional
        If True, plots a bar chart of outlier counts per numerical feature.

    Returns
    -------
    outlier_mask : np.ndarray of bool
        Boolean matrix (n_samples, n_num_features), True where a value is an outlier.
    outlier_counts : np.ndarray of int
        Number of outliers per numerical feature.
    """
    X_num = X[:, num_mask]
    n, d = X_num.shape
    outlier_mask = np.zeros((n, d), dtype=bool)

    for j in range(d):
        col = X_num[:, j]
        col_no_nan = col[~np.isnan(col)]
        if len(col_no_nan) == 0:
            continue
        q1 = np.percentile(col_no_nan, 25)
        q3 = np.percentile(col_no_nan, 75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask = (col < lower) | (col > upper)
        outlier_mask[:, j] = np.where(np.isnan(col), False, mask)

    outlier_counts = np.sum(outlier_mask, axis=0)

    if plot:
        plt.figure(figsize=(6, 4))
        plt.bar(np.arange(d), outlier_counts, edgecolor='black')
        plt.xlabel('Numerical feature index')
        plt.ylabel('Number of outliers')
        plt.title(f'Outliers per feature (threshold = {threshold} × IQR)')
        plt.tight_layout()
        plt.show()

    return outlier_mask, outlier_counts

def summarize_outliers(outlier_mask, num_mask, plot=False):
    """
    Summarizes the percentage of outliers per numerical feature.

    Parameters
    ----------
    outlier_mask : np.ndarray of bool
        Boolean matrix (n_samples, n_num_features)
    num_mask : np.ndarray of bool
        Boolean mask indicating numerical features
    plot : bool, optional
        If True, plots the percentage of outliers per numerical feature.

    Returns
    -------
    outlier_pctg : np.ndarray
        1D array of outlier percentages per numerical feature.
    """
    n = outlier_mask.shape[0]
    outlier_counts = np.sum(outlier_mask, axis=0)
    outlier_pctg = (outlier_counts / n) * 100

    if plot:
        d = len(outlier_pctg)
        plt.figure(figsize=(6, 4))
        plt.bar(np.arange(d), outlier_pctg, edgecolor='black')
        plt.xlabel('Numerical feature index')
        plt.ylabel('Outlier percentage (%)')
        plt.title('Percentage of outliers per numerical feature')
        plt.tight_layout()
        plt.show()

    return outlier_pctg