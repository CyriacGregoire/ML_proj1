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

def build_poly_multi(X, degree, num_mask=None):
    """
    Create polynomial features up to the given degree for numerical columns,
    excluding degree 0 (no bias column).

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data matrix.
    degree : int
        Maximum polynomial degree. Powers from 1 up to `degree` are generated.
    num_mask : ndarray of bool, optional
        Mask for numerical columns. If None, all are treated as numerical.

    Returns
    -------
    X_poly : ndarray
        Feature matrix including:
            - polynomial powers of numerical features (no bias term)
            - unchanged categorical features
    """
    n_samples, n_features = X.shape
    if num_mask is None:
        num_mask = np.ones(n_features, dtype=bool)

    poly = []
    for j in range(n_features):
        if num_mask[j]:
            for p in range(1, degree + 1):
                poly.append((X[:, j] ** p).reshape(-1, 1))
        else:
            poly.append(X[:, j].reshape(-1, 1))
    return np.hstack(poly)


def stratified_k_fold(X, y, k=5, seed=42):
    np.random.seed(seed)
    folds = []
    # unique classes
    classes = np.unique(y)

    # indices per class
    class_indices = [np.where(y == c)[0] for c in classes]
    for idx in class_indices:
        np.random.shuffle(idx)

    # split each class indices into k roughly equal parts
    class_folds = [np.array_split(idx, k) for idx in class_indices]

    # combine class folds to form stratified folds
    for i in range(k):
        val_idx = np.concatenate([folds[i] for folds in class_folds])
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        folds.append((train_idx, val_idx))
    
    return folds

def cross_validation_grid_search(
    y, X,
    k=5,
    lambdas=[1e-5, 1e-3, 1e-2],
    pos_weights=[1.0, 3.0, 5.0],
    thresholds=[0.3, 0.5, 0.7],
    gamma=0.05,
    max_iter=5000,
    tol=1e-8,
    seed=42,
    verbose=False,
    cleaning_function=None
):
    """
    Perform k-fold cross-validation over a grid of hyperparameters
    (lambda_, pos_weight, threshold) using weighted logistic regression.

    Parameters
    ----------
    y : np.ndarray
        Target vector (0/1).
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    k : int
        Number of folds.
    lambdas, pos_weights, thresholds : list
        Hyperparameter values to test.
    gamma : float
        Learning rate (fixed).
    max_iter : int
        Maximum number of iterations for logistic regression.
    tol : float
        Tolerance for convergence.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        If True, prints training details.
    cleaning_function : callable or None
        Function to clean data inside each fold:
        (X_train, y_train, X_val, y_val) → (X_train_clean, y_train, X_val_clean, y_val)

    Returns
    -------
    best_params : dict
        Dictionary with best hyperparameters.
    best_f1 : float
        Best mean F1-score found.
    results : list of tuples
        (lambda_, pos_weight, threshold, mean_f1)
    """
    np.random.seed(seed)
    folds = stratified_k_fold(X, y, k, seed)
    results = []
    best_f1 = -1.0
    best_params = None

    total = len(lambdas) * len(pos_weights)
    run = 0

    for lambda_ in lambdas:
        for pos_weight in pos_weights:
            neg_weight = 1.0  # fixed
            run += 1
            print(f"\n=== Run {run}/{total} | λ={lambda_} | pos_w={pos_weight} ===")

            f1_scores = []

            for i, (train_idx, val_idx) in enumerate(folds):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                # --- Apply cleaning function per fold ---
                if cleaning_function is not None:
                    X_train, y_train, X_val, y_val = cleaning_function(X_train, y_train, X_val, y_val)
                # ---------------------------------------

                # Train model
                loss, w = logistic_regression_weighted_gd(
                    y_train, X_train,
                    lambda_=lambda_,
                    gamma=gamma,
                    pos_weight=pos_weight,
                    neg_weight=neg_weight,
                    max_iter=max_iter,
                    tol=tol,
                    verbose=verbose
                )

                # Evaluate all thresholds
                for th in thresholds:
                    y_pred, _ = predict_with_threshold(X_val, w, threshold=th)
                    f1 = f1_score(y_val, y_pred)
                    f1_scores.append((th, f1))
                    print("For lambda", lambda_, "threshold", th, "pos_weight", pos_weight, "gets f1", f1)

            # Average F1 per threshold
            for th in thresholds:
                th_f1s = [f1 for t, f1 in f1_scores if t == th]
                mean_f1 = np.mean(th_f1s)
                results.append((lambda_, pos_weight, th, mean_f1))

                if mean_f1 > best_f1:
                    best_f1 = mean_f1
                    best_params = dict(lambda_=lambda_, pos_weight=pos_weight, threshold=th)
                    print(f"New best F1={best_f1:.4f} (λ={lambda_}, pos_w={pos_weight}, th={th})")

    # Sort results by descending mean F1
    results.sort(key=lambda x: x[3], reverse=True)

    print("\n=== Grid Search Summary ===")
    print(f"Best mean F1={best_f1:.4f}")
    print(f"Best params: λ={best_params['lambda_']}, pos_w={best_params['pos_weight']}, "
          f"threshold={best_params['threshold']}")

    return best_params, best_f1, results