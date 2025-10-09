import numpy as np
import matplotlib.pyplot as plt

#Loss Functions

def mse_loss(y, tx, w):
    error = y - tx @ w
    return np.mean(error ** 2) / 2

def sigmoid(t):
    """Numerically stable sigmoid."""
    return np.where(t >= 0,
                    1 / (1 + np.exp(-t)),
                    np.exp(t) / (1 + np.exp(t)))

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
        
    return ws[-1], loss[-1]

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
        
    return ws[-1], loss[-1]

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
          
def penalized_logistic_regression(y, tx, w, lambda_):
    loss = logistic_loss(y, tx, w)
    grad = compute_gradient_logistic(y, tx, w)
    grad += 2* lambda_ * w  
    return loss, grad

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):

    loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * grad
    return loss, w
    
def logistic_regression_penalized_gradient_descent_demo(
    y, x, max_iter=10000, gamma=0.5, lambda_=0.0005, threshold=1e-8
):
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros(tx.shape[1])

    for iter in range(max_iter):
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        losses.append(loss)

        if iter % 100 == 0:
            print(f"Iteration {iter:5d}, loss = {loss:.6f}")

        # convergence check
        if len(losses) > 1 and abs(losses[-1] - losses[-2]) < threshold:
            print(f"Converged at iteration {iter}")
            break

    return loss, w
            

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
    return w, rmse


