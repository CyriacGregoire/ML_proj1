import numpy as np
import matplotlib.pyplot as plt

def remove_nan_features(X, threshold=0.3):
    """
    Remove columns (features) with more than threshold proportion of NaN values.
    """
    nan_per_feature = np.sum(np.isnan(X), axis=0)
    keep_mask = nan_per_feature < threshold * X.shape[0]
    X_clean = X[:, keep_mask]
    return X_clean, keep_mask

def impute_numerical(X, num_mask, medians=None):
    """
    Imputes NaN values in numerical features with their medians.
    If predefined medians are provided, they are used instead.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_samples, n_features)
    num_mask : np.ndarray of bool
        Boolean mask where True means the feature is numerical.
    medians : np.ndarray or None, optional
        Predefined medians for numerical features (used if not None).

    Returns
    -------
    X_imp : np.ndarray
        Copy of X with NaNs in numerical features imputed.
    medians : np.ndarray
        Median values used for each numerical feature.
    """
    X_imp = X.copy().astype(float)
    idx_num = np.where(num_mask)[0]

    # If medians are not provided, compute them from X
    if medians is None:
        medians = np.nanmedian(X_imp[:, idx_num], axis=0)

    # Replace NaNs with the corresponding median
    for i, j in enumerate(idx_num):
        col = X_imp[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            col[nan_mask] = medians[i]
            X_imp[:, j] = col

    return X_imp, medians

def impute_categorical(X, cat_mask, modes=None):
    """
    Imputes NaN values in categorical features with the mode (most frequent value).
    If predefined modes are provided, they are used instead.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_samples, n_features)
    cat_mask : np.ndarray of bool
        Boolean mask where True means the feature is categorical.
    modes : np.ndarray or None, optional
        Predefined modes for categorical features (used if not None).

    Returns
    -------
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


def stratified_three_way_split(X, y, val_ratio=0.15, test_ratio=0.15, seed=0):
    """
    Returns a dataset split into train, validation , and test set given the ratios,
    keeping the same ratio of class balance on each set. It uses a fixed seed for 
    reproductibility

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    val_ratio : TYPE, optional
        DESCRIPTION. The default is 0.15.
    test_ratio : TYPE, optional
        DESCRIPTION. The default is 0.15.
    seed : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    np.random.seed(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    np.random.shuffle(idx_pos)
    np.random.shuffle(idx_neg)

    n_pos, n_neg = len(idx_pos), len(idx_neg)
    n_val_pos = int(n_pos * val_ratio)
    n_test_pos = int(n_pos * test_ratio)
    n_val_neg = int(n_neg * val_ratio)
    n_test_neg = int(n_neg * test_ratio)

    val_idx = np.concatenate([idx_pos[:n_val_pos], idx_neg[:n_val_neg]])
    test_idx = np.concatenate([
        idx_pos[n_val_pos:n_val_pos + n_test_pos],
        idx_neg[n_val_neg:n_val_neg + n_test_neg]
    ])
    train_idx = np.concatenate([
        idx_pos[n_val_pos + n_test_pos:],
        idx_neg[n_val_neg + n_test_neg:]
    ])

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx]
    )

def one_hot_encode(X, cat_mask, categories=None, drop_first=True):
    """
    One-hot encodes categorical features.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_samples, n_features)
    cat_mask : np.ndarray of bool
        Boolean mask where True means the feature is categorical.
    categories : list of np.ndarray or None
        List containing the unique values for each categorical feature.
        If None, categories are computed from X (use this for training).
    drop_first : bool
        If True, drops the first category to avoid collinearity.

    Returns
    -------
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


def standardize_features(X, num_mask, means=None, stds=None):
    """
    Standardizes numerical features (mean 0, std 1).
    Non-numerical (categorical / one-hot) features are left unchanged.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_samples, n_features)
    num_mask : np.ndarray of bool
        Boolean mask where True means the feature is numerical.
    means : np.ndarray or None
        Optional predefined means for numerical features (used for val/test)
    stds : np.ndarray or None
        Optional predefined stds for numerical features (used for val/test)

    Returns
    -------
    X_std : np.ndarray
        Standardized copy of X.
    means : np.ndarray
        Means used for numerical features.
    stds : np.ndarray
        Standard deviations used for numerical features.
    """
    X_std = X.copy().astype(float)
    idx_num = np.where(num_mask)[0]

    if means is None or stds is None:
        means = np.mean(X_std[:, idx_num], axis=0)
        stds = np.std(X_std[:, idx_num], axis=0)
        stds[stds == 0] = 1.0  # avoid division by zero

    X_std[:, idx_num] = (X_std[:, idx_num] - means) / stds
    return X_std, means, stds



def feature_target_correlation(X, y):
    """
    Computes Pearson correlation between each feature and the target y.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_samples, n_features)
    y : np.ndarray
        Target vector (n_samples,)

    Returns
    -------
    corr : np.ndarray
        1D array with correlation values between each feature and y.
    """
    Xc = X - np.mean(X, axis=0)
    yc = y - np.mean(y)
    num = np.sum(Xc * yc[:, None], axis=0)
    denom = np.sqrt(np.sum(Xc ** 2, axis=0) * np.sum(yc ** 2))
    corr = np.zeros(X.shape[1])
    valid = denom > 0
    corr[valid] = num[valid] / denom[valid]
    return corr

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

def sigmoid(z):
    """Numerically stable sigmoid."""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def weighted_logistic_loss(y, tx, w, lambda_=0.0, pos_weight=1.0, neg_weight=1.0):
    """
    Weighted (and optionally penalized) logistic loss.
    """
    p = sigmoid(tx @ w)
    eps = 1e-15  # avoid log(0)

    # weights per sample
    sample_weights = np.where(y == 1, pos_weight, neg_weight)

    # weighted average loss
    loss = -np.sum(sample_weights * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))) / np.sum(sample_weights)
    
    # no regularization term added to returned loss (for monitoring only)
    return float(loss)


def weighted_gradient_logistic(y, tx, w, lambda_=0.0, pos_weight=1.0, neg_weight=1.0):
    """
    Gradient of the weighted logistic loss with L2 penalty.
    """
    p = sigmoid(tx @ w)
    sample_weights = np.where(y == 1, pos_weight, neg_weight)
    error = sample_weights * (p - y)
    grad = (tx.T @ error) / np.sum(sample_weights)
    grad[1:] += 2 * lambda_ * w[1:]  # don't regularize bias
    return grad.ravel()

def logistic_regression_weighted_gd(
    y, x, lambda_=1e-3, gamma=0.05, pos_weight=1.0, neg_weight=1.0,
    max_iter=10000, tol=1e-8, clip_grad=10.0, verbose=True
):
    """
    Logistic regression with class weights and L2 regularization.
    Returns (loss, w).
    """
    tx = np.c_[np.ones((x.shape[0], 1)), x]
    w = np.zeros(tx.shape[1])
    losses = []

    for it in range(max_iter):
        grad = weighted_gradient_logistic(y, tx, w, lambda_, pos_weight, neg_weight)
        grad_norm = np.linalg.norm(grad)
        if grad_norm > clip_grad:
            grad *= clip_grad / grad_norm

        loss = weighted_logistic_loss(y, tx, w, lambda_, pos_weight, neg_weight)
        losses.append(loss)

        w -= gamma * grad

        if it > 0 and abs(losses[-1] - losses[-2]) < tol:
            if verbose:
                print(f"Converged at iteration {it}")
            break

        if verbose and it % 100 == 0:
            print(f"Iter {it:5d} | Loss = {loss:.6f} | GradNorm = {grad_norm:.4f}")

    return losses[-1], w

def predict_with_threshold(x, w, threshold=0.5):
    tx = np.c_[np.ones((x.shape[0], 1)), x]
    probs = sigmoid(tx @ w)
    preds = (probs >= threshold).astype(int)
    return preds, probs

def f1_score(y_true, y_pred):
    """
    Compute F1 score using only NumPy.
    Works for binary classification (0/1).
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp + 1e-15)
    recall = tp / (tp + fn + 1e-15)
    f1 = 2 * precision * recall / (precision + recall + 1e-15)
    return f1

def accuracy_numpy(y_true, y_pred):
    """
    Compute accuracy.
    """
    return np.mean(y_true == y_pred)


# --- Full evaluation wrapper ---
def evaluate_model(y_true, X, w, threshold=0.5):
    """
    Evaluate trained logistic regression on a dataset.
    Returns accuracy and F1 score.
    """
    preds, probs = predict_with_threshold(X, w, threshold=threshold)
    acc = accuracy_numpy(y_true, preds)
    f1 = f1_score(y_true, preds)
    print(f" Accuracy: {acc*100:.2f}%")
    print(f" F1 Score: {f1:.4f}")
    return acc, f1

def grid_search(
    y_train, X_train,
    y_val, X_val,
    pos_weights=[1, 3, 5, 9],
    lambdas=[1e-5, 1e-3, 1e-2, 1e-1],
    thresholds=[0.3, 0.5, 0.7],
    max_iter=10000,
    gamma=0.05
):
    """
    Grid search for weighted penalized logistic regression.
    Returns: best_params, best_f1, results_list
    """
    best_f1 = -1
    best_params = None
    results = []

    total = len(pos_weights) * len(lambdas) * len(thresholds)
    run = 0

    for pw in pos_weights:
        for lam in lambdas:
            run += 1
            print(f"\n=== Run {run}/{total//len(thresholds)} (pos_weight={pw}, lambda_={lam}) ===")

            try:
                # Train model
                loss, w = logistic_regression_weighted_gd(
                    y_train, X_train,
                    lambda_=lam,
                    gamma=gamma,
                    pos_weight=pw,
                    neg_weight=1.0,
                    max_iter=max_iter,
                    verbose=False
                )

                # Skip invalid runs
                if np.isnan(loss) or np.isinf(loss) or loss > 10:
                    print(f"Invalid loss ({loss:.4f}), skipping.")
                    continue

                # Evaluate all thresholds for this model
                for th in thresholds:
                    preds, _ = predict_with_threshold(X_val, w, threshold=th)
                    f1 = f1_score(y_val, preds)
                    results.append((pw, lam, th, f1))

                    if f1 > best_f1:
                        best_f1 = f1
                        best_params = (pw, lam, th)
                        print(f"New best F1 = {best_f1:.4f} (threshold = {th:.2f})")

            except Exception as e:
                print(f"Error for pos_weight={pw}, lambda_={lam}: {e}")
                continue

    # Sort results by F1 descending
    results.sort(key=lambda t: t[3], reverse=True)

    print("\n=== Grid Search Complete ===")
    if best_params:
        print(f"Best F1 = {best_f1:.4f} at pos_weight={best_params[0]}, λ={best_params[1]}, threshold={best_params[2]}")
    else:
        print("No valid runs completed.")

    return best_params, best_f1, results



def confusion_matrix_numpy(y_true, y_pred, plot=True):
    """
    Compute and optionally plot the confusion matrix using NumPy.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_pred : np.ndarray
        Predicted binary labels (0 or 1)
    plot : bool, optional
        If True, displays the confusion matrix as a heatmap

    Returns
    -------
    cm : np.ndarray
        2x2 confusion matrix [[TN, FP], [FN, TP]]
    """

    # Ensure binary
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    cm = np.array([[tn, fp], [fn, tp]])

    print("Confusion Matrix:")
    print(cm)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    if plot:
        plt.figure(figsize=(4, 3))
        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.xticks([0, 1], ["0", "1"])
        plt.yticks([0, 1], ["0", "1"])

        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12)

        plt.tight_layout()
        plt.show()

    return cm


def impute_categorical_missing_code(X, cat_mask, missing_code=-1.0):
    """
    Replace NaNs in *categorical* columns with a special code (default -1),
    leaving numeric columns unchanged.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Full feature matrix (numeric + categorical).
    cat_mask : np.ndarray of bool, shape (n_features,)
        True for categorical columns, False for numeric ones.
    missing_code : float, default=-1.0
        Code used to represent 'Missing' category.

    Returns
    -------
    X_imputed : np.ndarray, same shape as X
        Copy of X where NaNs in categorical columns are replaced by `missing_code`.
    """
    Xf = np.array(X, dtype=np.float64, copy=True)
    cat_idx = np.where(cat_mask)[0]

    for j in cat_idx:
        col = Xf[:, j]
        nan_mask = np.isnan(col)
        if np.any(nan_mask):
            col[nan_mask] = missing_code
            Xf[:, j] = col

    return Xf