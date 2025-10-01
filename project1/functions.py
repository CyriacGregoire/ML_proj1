import numpy as np
import matplotlib.pyplot as plt

def mse_loss(y, tx, w):
    error = y - tx @ w
    return np.mean(error ** 2) / 2

def compute_gradient(y, tx, w):
    N = y.shape[0]
    error = y - tx @ w
    return -(tx.T @ error) / N
    
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
    
    
def least_squares(y, tx):
  
    
def ridge_regression(y, tx, lambda_):
    
     
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    
    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):