import numpy as np
import matplotlib.pyplot as plt

def mse_loss(y, tx, w):
    error = y - tx @ w
    return np.mean(error ** 2) / 2

def compute_gradient(y, tx, w):
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
        
    return losses, ws

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
        
    return losses, ws
        
def least_squares(y, tx):
    N = np.shape(y)[0]
    D = np.shape(tx)[1]
    w = np.zeros(D)

    XTX = np.transpose(tx)@tx
    XTX_inv = np.linalg.inv(XTX)
    w = (XTX_inv@np.transpose(tx))@y
    L = (np.transpose(y-tx@w)@(y-tx@w))/(2*N)

    return w,L

print('fwe')
least_squares(np.array([1,1]),np.array([[2,1],[1,7]]))
    
def ridge_regression(y, tx, lambda_):

    N = np.shape(y)[0]
    D = np.shape(tx)[1]
    w = np.zeros(D)
    lambda_prime = lambda_*2*N

    XTX = np.transpose(tx)@tx    
    w = np.linalg.inv(XTX + lambda_prime*np.identity(D))@np.transpose(tx)@y
    
    return w
     
# def logistic_regression(y, tx, initial_w, max_iters, gamma):
    
    
# def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):