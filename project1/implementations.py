import numpy as np
import matplotlib.pyplot as plt

def mse_loss(y, tx, w):
    error = y - tx @ w
    return np.mean(error ** 2) / 2

def compute_gradient_mse(y, tx, w):
    N = y.shape[0]
    error = y - tx @ w
    return -(tx.T @ error) / N

def compute_gradient_stochastic(y_i, x_i, w):
    error = y_i - x_i @ w        
    return -error * x_i 
    
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)    
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

     
# def logistic_regression(y, tx, initial_w, max_iters, gamma):
    
    
# def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):