import numpy as np
from helpers import model_accuracy, f1_score

# Logistic Regression Functions


def sigmoid(z):
    
    "Compute the sigmoid function in a numerically stable way."

    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def predict_logistic(x, w, threshold=0.5):

    """
    Args:
        x: Input data, shape (N, D)
        w: Weights, shape (D + 1,)
        threshold: Classification threshold
    Returns:
        preds: Predicted class labels (0 or 1), shape (N,)
    """

    tx = np.c_[np.ones((x.shape[0], 1)), x]
    probs = sigmoid(tx @ w)
    preds = (probs >= threshold).astype(int)
    return preds

def evaluate_logistic_model(y_true, X, w, threshold=0.5):
    """
    Args:
        y_true: True labels, shape (N,)
        X: Input data, shape (N, D)
        w: Weights, shape (D + 1,)
        threshold: Classification threshold

    Evaluate trained logistic regression on a dataset.
    Returns accuracy and F1 score.
    """
    preds = predict_logistic(X, w, threshold=threshold)
    acc = model_accuracy(y_true, preds)
    f1 = f1_score(y_true, preds)
    print(f" Accuracy: {acc*100:.2f}%")
    print(f" F1 Score: {f1:.4f}")
    return acc, f1


# Logistic Regression with Gradient Descent

def logistic_loss(y, tx, w):
    """Stable logistic loss computation."""
    eps = 1e-15
    pred = sigmoid(tx @ w)
    pred = np.clip(pred, eps, 1 - eps)
    loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    
    return float(loss)

def compute_gradient_logistic(y, tx, w):
    pred = sigmoid(tx @ w)               
    grad = tx.T @ (pred - y) / y.shape[0]
    return grad
    

def logistic_regression_penalized_gradient_descent(
        y, x, max_iter=10000, gamma=0.5, lambda_=1e-3, threshold=1e-8):
    
    losses = []
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

# K-fold Cross-Validation for Logistic Regression

def kfold_logistic_ridge(
        X, y, process_data, k=5, gamma=0.5, lambda_=1e-3, alpha = 0.7 , threshold=1e-8, random_state=True):
    
    """
    Args:
        X: Input data, shape (N, D)
        y: Labels, shape (N,)
        process_data(X_train, X_val, y_train, y_val): 
            Data preprocessing function with returns: 
            X_train_process, X_val_process, y_train_process, y_val_process
        k: Number of folds
        gamma: Learning rate
        lambda_: Regularization parameter
        alpha: threshold probability for classification
        threshold: Convergence threshold
        random_state: Seed for reproducibility"""


    if random_state is not None:
        np.random.seed(random_state)

    n = len(y)
    indices = np.random.permutation(n)  # Shuffle data
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[:n % k] += 1

    current = 0
    f1_scores = []
    accuracies = []
    current_batch = 1


    for fold_size in fold_sizes:
        print("Start cleaning batch", current_batch, "out of", k)
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Preprocess data

        X_train, X_val, y_train, y_val = process_data(X_train, X_val, y_train, y_val)

        print("Cleaning of batch", current_batch, "done. Stating the model training.")

        # training the model and computing error

        loss, w = logistic_regression_penalized_gradient_descent(y_train, X_train, lambda_=lambda_, gamma=gamma, threshold=threshold)
        acc, f1 = evaluate_logistic_model(y_val, X_val, w, threshold=alpha)
        accuracies.append(acc)
        f1_scores.append(f1)       
        current += fold_size
        current_batch += 1 
        
    mean_accuracy = np.mean(accuracies)
    mean_f1 = np.mean(f1_scores)    
    return mean_accuracy, mean_f1

    

