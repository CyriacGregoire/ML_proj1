import numpy as np

# Loss Functions
#############################################################################################################

def mse_loss(y, tx, w):
    """
    Compute the mean squared error (MSE) loss.

    Parameters
    ----------
    y : ndarray of shape (N,)
        Target values.
    tx : ndarray of shape (N, D)
        Input data matrix.
    w : ndarray of shape (D,)
        Weight vector.

    Returns
    -------
    float
        The MSE loss value.
    """
    error = y - tx @ w
    return np.mean(error ** 2) / 2


def sigmoid(z):
    """
    Apply the sigmoid function in a numerically stable way.

    Parameters
    ----------
    z : ndarray
        Input array (can be scalar, vector or matrix).

    Returns
    -------
    ndarray
        Output after applying the sigmoid function elementwise.
    """
    # Clip z to avoid overflow in exp()
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def logistic_loss(y, tx, w):
    """
    Compute the logistic loss (negative log-likelihood).

    Parameters
    ----------
    y : ndarray of shape (N,)
        Binary target values (0 or 1).
    tx : ndarray of shape (N, D)
        Input data matrix.
    w : ndarray of shape (D,)
        Weight vector.

    Returns
    -------
    float
        The logistic loss value.
    """
    eps = 1e-15
    pred = sigmoid(tx @ w)
    pred = np.clip(pred, eps, 1 - eps)
    loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    return float(loss)


# Gradient Computation
#############################################################################################################

def compute_gradient_mse(y, tx, w):
    """
    Compute the gradient of the MSE loss.

    Parameters
    ----------
    y : ndarray of shape (N,)
        Target values.
    tx : ndarray of shape (N, D)
        Input data matrix.
    w : ndarray of shape (D,)
        Weight vector.

    Returns
    -------
    ndarray of shape (D,)
        Gradient of the MSE loss.
    """
    N = y.shape[0]
    error = y - tx @ w
    return -(error.T @ tx) / N


def compute_gradient_stochastic(y_i, x_i, w):
    """
    Compute the stochastic gradient for one data sample.

    Parameters
    ----------
    y_i : float
        Single target value.
    x_i : ndarray of shape (D,)
        Single input data sample.
    w : ndarray of shape (D,)
        Weight vector.

    Returns
    -------
    ndarray of shape (D,)
        Stochastic gradient for the sample.
    """
    error = y_i - x_i @ w        
    return -error * x_i 


def compute_gradient_logistic(y, tx, w):
    """
    Compute the gradient of the logistic loss.

    Parameters
    ----------
    y : ndarray of shape (N,)
        Binary target values (0 or 1).
    tx : ndarray of shape (N, D)
        Input data matrix.
    w : ndarray of shape (D,)
        Weight vector.

    Returns
    -------
    ndarray of shape (D,)
        Gradient of the logistic loss.
    """
    pred = sigmoid(tx @ w) 
    grad = (tx.T @ (pred - y)) / y.shape[0]
    return grad
    

# Gradient Descent Methods
#############################################################################################################

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Perform gradient descent using the MSE loss.

    Parameters
    ----------
    y : ndarray of shape (N,)
        Target values.
    tx : ndarray of shape (N, D)
        Input data matrix.
    initial_w : ndarray of shape (D,)
        Initial weight vector.
    max_iters : int
        Maximum number of iterations.
    gamma : float
        Learning rate.

    Returns
    -------
    w : ndarray of shape (D,)
        Final weights after training.
    loss : float
        Final MSE loss value.
    """
    w = initial_w
    loss = mse_loss(y, tx, w)
    
    for n_iter in range(max_iters):
        gradient = compute_gradient_mse(y, tx, w)    
        w = w - gamma * gradient
        loss = mse_loss(y, tx, w)
        
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Perform stochastic gradient descent using the MSE loss.

    Parameters
    ----------
    y : ndarray of shape (N,)
        Target values.
    tx : ndarray of shape (N, D)
        Input data matrix.
    initial_w : ndarray of shape (D,)
        Initial weight vector.
    max_iters : int
        Maximum number of iterations.
    gamma : float
        Learning rate.

    Returns
    -------
    w : ndarray of shape (D,)
        Final weights after training.
    loss : float
        Final MSE loss value.
    """
    w = initial_w
    loss = mse_loss(y, tx, w)
    N = y.shape[0]
 
    for n_iter in range(max_iters):
        i = np.random.randint(0, N)
        x_i, y_i = tx[i], y[i]
        gradient = compute_gradient_stochastic(y_i, x_i, w) 
        w = w - gamma * gradient
        loss = mse_loss(y, tx, w)
        
    return w, loss


# Logistic Regression
#############################################################################################################

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Run logistic regression using gradient descent.

    Parameters
    ----------
    y : ndarray of shape (N,)
        Binary target values (0 or 1).
    tx : ndarray of shape (N, D)
        Input data matrix.
    initial_w : ndarray of shape (D,)
        Initial weight vector.
    max_iters : int
        Maximum number of iterations.
    gamma : float
        Learning rate.

    Returns
    -------
    w : ndarray of shape (D,)
        Final weights after training.
    loss : float
        Final logistic loss value.
    """
    loss = logistic_loss(y, tx, initial_w)
    w = initial_w

    for iter in range(max_iters):
        grad = compute_gradient_logistic(y, tx, w)
        loss = logistic_loss(y, tx, w)
        w = w - gamma * grad

        if iter % 100 == 0:
            print(f"Current iteration={iter}, loss={loss}")

    loss = np.array([loss])[0]
    return w, loss
          

# Regularized Logistic Regression
#############################################################################################################

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Run logistic regression with L2 regularization.

    Parameters
    ----------
    y : ndarray of shape (N,)
        Binary target values (0 or 1).
    tx : ndarray of shape (N, D)
        Input data matrix.
    lambda_ : float
        Regularization strength.
    initial_w : ndarray of shape (D,)
        Initial weight vector.
    max_iters : int
        Maximum number of iterations.
    gamma : float
        Learning rate.

    Returns
    -------
    w : ndarray of shape (D,)
        Final weights after training.
    loss : float
        Final logistic loss value.
    """
    losses = []
    loss = logistic_loss(y, tx, initial_w)
    w = initial_w

    for iter in range(max_iters):
        loss = logistic_loss(y, tx, w)
        grad = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w  
        w = w - gamma * grad
        losses.append(loss)

        if iter % 100 == 0:
            print(f"Iteration {iter:5d}, loss = {loss:.6f}")

    loss = np.array([loss])[0]
    return w, loss


# Explicit Solutions
#############################################################################################################

def least_squares(y, tx):
    """
    Compute the least squares solution using the normal equations.

    Parameters
    ----------
    y : ndarray of shape (N,)
        Target values.
    tx : ndarray of shape (N, D)
        Input data matrix.

    Returns
    -------
    w : ndarray of shape (D,)
        Optimal weights.
    loss : float
        Final MSE loss value.
    """
    A = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    mse = mse_loss(y, tx, w)
    return w, mse


def ridge_regression(y, tx, lambda_):
    """
    Compute the ridge regression solution using the normal equations.

    Parameters
    ----------
    y : ndarray of shape (N,)
        Target values.
    tx : ndarray of shape (N, D)
        Input data matrix.
    lambda_ : float
        Regularization strength.

    Returns
    -------
    w : ndarray of shape (D,)
        Optimal weights.
    loss : float
        Final MSE loss value.
    """
    D = tx.shape[1]
    N = y.shape[0]
    I = np.eye(D)
    lambda_prime = lambda_ * 2 * N
    A = tx.T @ tx + lambda_prime * I
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    loss = mse_loss(y, tx, w)
    return w, loss




