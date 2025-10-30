import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm
from scipy.stats import chi2

#Loss Functions

def mse_loss(y, tx, w):
    error = y - tx @ w
    return np.mean(error ** 2) / 2

def sigmoid(z):

    # Clip z to avoid overflow in exp()
    z = np.clip(z, -500, 500)  # exp(709) is close to float64 max (~1e308)

    return 1.0 / (1.0 + np.exp(-z))

def logistic_loss(y, tx, w):
    """Stable logistic loss computation."""
    eps = 1e-15
    pred = sigmoid(tx @ w)
    pred = np.clip(pred, eps, 1 - eps)
    loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    
    return float(loss)

#Compute Gradients

def compute_gradient_mse(y, tx, w):
    N = y.shape[0]
    error = y - tx @ w
    return -(tx.T @ error) / N

def compute_gradient_stochastic(y_i, x_i, w):
    error = y_i - x_i @ w        
    return -error * x_i 

def compute_gradient_logistic(y, tx, w):
    pred = sigmoid(tx @ w)               
    grad = tx.T @ (pred - y) / y.shape[0]
    return grad
    

#Gradient Descent

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        gradient = compute_gradient_mse(y, tx, w)    
        loss = mse_loss(y, tx, w)
        
        w = w - gamma * gradient
        
        ws.append(w)
        losses.append(loss)
        
    return ws[-1], losses[-1]

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    N = y.shape[0]
 
    for n_iter in range(max_iters):
        i = np.random.randint(0, N)
        x_i, y_i = tx[i], y[i]
        
        gradient = compute_gradient_stochastic(y_i, x_i, w) 
        loss = mse_loss(y, tx, w)
        
        w = w - gamma * gradient
        
        ws.append(w)
        losses.append(loss)
        
    return ws[-1], losses[-1]

def learning_by_gradient_descent_logistic(y, tx, w, gamma):
    grad = compute_gradient_logistic(y, tx, w)
    loss = logistic_loss(y, tx, w)
    w = w - gamma * grad

    return loss, w

def logistic_regression_gradient_descent(y, x):
    # init parameters
    max_iter = 10000
    threshold = 1e-8
    gamma = 0.5
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent_logistic(y, tx, w, gamma)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
          

# Logistic Regression with Penalization


def logistic_regression_penalized_gradient_descent(
    y, x, max_iter=10000, gamma=0.5, lambda_=1e-3, threshold=1e-8
):
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros(tx.shape[1])

    for iter in range(max_iter):
        loss = logistic_loss(y, tx, w) + lambda_ * w.T@w
        grad = compute_gradient_logistic(y, tx, w) + 2* lambda_ * w  
        w = w - gamma * grad
        losses.append(loss)

        if iter % 100 == 0:
            print(f"Iteration {iter:5d}, loss = {loss:.6f}")

        # convergence check
        if len(losses) > 1 and abs(losses[-1] - losses[-2]) < threshold:
            print(f"Converged at iteration {iter}")
            break

    return loss, w
###########################################################################################################################################################

def logistic_regression(y, x):
    # init parameters
    max_iter = 10000
    threshold = 1e-8
    gamma = 0.5
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent_logistic(y, tx, w, gamma)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    w, losses[-1]
          

# Logistic Regression with Penalization


def reg_logistic_regression(
    y, x, max_iter=10000, gamma=0.5, lambda_=1e-3, threshold=1e-8
):
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros(tx.shape[1])

    for iter in range(max_iter):
        loss = logistic_loss(y, tx, w) + lambda_ * w.T@w
        grad = compute_gradient_logistic(y, tx, w) + 2* lambda_ * w  
        w = w - gamma * grad
        losses.append(loss)

        if iter % 100 == 0:
            print(f"Iteration {iter:5d}, loss = {loss:.6f}")

        # convergence check
        if len(losses) > 1 and abs(losses[-1] - losses[-2]) < threshold:
            print(f"Converged at iteration {iter}")
            break

    return w, loss

###########################################################################################################################################################
            
def predict_logistic(X, w, limit=0.5):
    """
    Computes the model prediction for a trained w parameters
    X should be normalized, without NaNs and without intercept column
    limit is the threshold to decide between class 0 and 1

    returns: a vector of 0/1 predictions
    """

    tx = np.hstack([np.ones((X.shape[0], 1)), X])
    return (sigmoid(tx @ w) >= limit).astype(int)

def predict_ridge(X, w, limit=0.5):
    """
    Computes the model prediction for a trained w parameters
    X should be normalized, without NaNs and without intercept column
    limit is the threshold to decide between class 0 and 1

    returns: a vector of 0/1 predictions
    """

    tx = np.hstack([np.ones((X.shape[0], 1)), X])
    return (tx @ w >= limit).astype(int)

def kfold_logistic_ridge(
        X, y, k=5, gamma=0.5, lambda_=1e-3, threshold=1e-8, random_state=None):
    """
    Performs K-Fold Cross Validation for logistic regression.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features) with Nan values and unbalanced data
    all the features will be used in this model (otherwise delete befor calling the function)
    y : np.ndarray, shape (n_samples,) with binary labels (0/1)
    k : int
        Number of folds.
    lam : float
        Regularization strength.
    lr : float
        Learning rate.
    epochs : int
        Number of gradient descent epochs.
    batch_size : int
        Batch size for training.
    random_state : int or None
        For reproducible shuffling.

    Returns
    -------
    mean_accuracy : float
        Average validation accuracy across folds.
    mean_dispersions_pred : float
        Average predicted dispersion across folds.
    mean_dispersions_true : float
        Average true dispersion across folds.
    w : np.ndarray, shape (n_features + 1,)
        Model parameters from the last fold (arbitrary fold).
    """
    if random_state is not None:
        np.random.seed(random_state)

    n = len(y)
    indices = np.random.permutation(n)  # Shuffle data
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[:n % k] += 1

    current = 0
    scores = []
    dispersions_pred = []
    dispersions_true = []
    current_batch = 1
    w = []

    for fold_size in fold_sizes:
        print("Start cleaning batch", current_batch, "out of", k)
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # clean and balance data

        X_train, X_val, y_train, y_val = clean_data(X_train, X_val, y_train, y_val)

        print("Cleaning of batch", current_batch, "done. Stating the model training.")

        # training the model and computing error

        loss, w = logistic_regression_penalized_gradient_descent(y_train, X_train, lambda_=lambda_, gamma=gamma, threshold=threshold)
        preds = predict_logistic(X_val, w)
        acc = np.mean(preds == y_val)
        dispersion_pred = np.sum(preds) / preds.shape[0]
        dispersion_true = np.sum(y_val) / y_val.shape[0]
        dispersions_pred.append((dispersion_pred))
        dispersions_true.append((dispersion_true))

        print("Dispersion in validation set:", dispersion_true)
        print("Dispersion in predictions:", dispersion_pred)
        scores.append(acc)

        current = stop
        print("Training done. Score for batch", current_batch, ":", acc)
        print()
        current_batch += 1

    return w, np.mean(scores), np.mean(dispersions_pred), np.mean(dispersions_true)


#Explicit Solutions
        
def least_squares(y, tx):
    A = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    mse = mse_loss(y, tx, w)
    return w, mse

def ridge_regression(y, tx, lambda_):
    D = tx.shape[1]
    N = y.shape[0]
    I = np.eye(D)
    lambda_prime = lambda_ * 2 * N
    A = tx.T @ tx + lambda_prime * I
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    rmse = np.sqrt(2 * mse_loss(y, tx, w))
    return rmse, w



def logistic_tvalues(X, w, lambda_=0.0):
    """
    Compute t-values for a penalized logistic regression model (NumPy only).

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Feature matrix (without intercept column).
    w : ndarray of shape (d + 1,) or (d + 1, 1)
        Estimated weights including intercept.
    lambda_ : float, optional
        L2 regularization strength (default 0.0).

    Returns
    -------
    t_values : ndarray of shape (d + 1,)
        t-values for each coefficient (including intercept).
    """
    n, d = X.shape

    # Ensure w is a column vector
    w = w.reshape(-1, 1)

    # Add intercept to X
    X_aug = np.hstack([np.ones((n, 1)), X])  # (n, d+1)

    # Predicted probabilities
    z = X_aug @ w
    p = 1 / (1 + np.exp(-z))

    # Diagonal weights
    W = (p * (1 - p)).flatten()

    # Hessian with L2 penalty (except intercept)
    penalty = lambda_ * np.eye(d + 1)
    penalty[0, 0] = 0.0
    H = X_aug.T @ (X_aug * W[:, None]) + penalty

    # Covariance matrix
    H_inv = np.linalg.inv(H)

    # Standard errors
    se = np.sqrt(np.diag(H_inv)).reshape(-1, 1)

    # t-values (same shape as w)
    t_values = (w / se).flatten()

    return t_values



def one_step_elimination_mask(w, H, alpha=0.05):
    """
    Single-step backward elimination returning a binary mask.
    
    Parameters:
    ------------
    w : np.ndarray, shape (p,)
        Model coefficients
    H : np.ndarray, shape (p,p)
        Hessian of the model
    alpha : float
        Significance level (for chi-square threshold)
    
    Returns:
    ---------
    mask : np.ndarray, shape (p,)
        Binary mask: 1 if feature kept, 0 if eliminated
    """
    p = len(w)
    mask = np.ones(p, dtype=int)
    threshold = chi2.ppf(1 - alpha, df=1)
    delta = (w ** 2) / np.diag(H)
    mask[delta < threshold] = 0
    
    return mask, delta

