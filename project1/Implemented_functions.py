import numpy as np

# Logistic Regression Functions

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


def evaluate_logistic_model(y_true, X, w, threshold=0.5, lambda_=1e-3):
    """
    Args:
        y_true: True labels, shape (N,)
        X: Input data, shape (N, D)
        w: Weights, shape (D + 1,)
        threshold: Classification threshold

    Evaluate trained logistic regression on a dataset.
    Returns accuracy and F1 score.

    
    """
    tx = np.c_[np.ones((X.shape[0], 1)), X]
    preds = predict_logistic(X, w, threshold=threshold)
    acc = np.mean(y_true == preds)
    f1 = f1_score(y_true, preds)
    loss = logistic_pen_loss(y_true, tx, w, lambda_)
    print(f" Accuracy: {acc*100:.2f}%")
    print(f" F1 Score: {f1:.4f}")
    return f1, acc, loss


# Logistic Regression with Gradient Descent

def logistic_pen_loss(y, tx, w, lambda_):
    """Stable logistic loss computation."""
    eps = 1e-15
    pred = sigmoid(tx @ w)
    pred = np.clip(pred, eps, 1 - eps)
    loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred)) + lambda_ * w.T@w
    
    return float(loss)

def compute_gradient_logistic_pen(y, tx, w, lambda_):
    pred = sigmoid(tx @ w)               
    grad = tx.T @ (pred - y) / y.shape[0]
    return grad + 2* lambda_ * w 
    

def logistic_regression_penalized_gradient_descent(
        y, x, max_iter=10000, gamma=0.5, lambda_=1e-3, threshold=1e-8):
    
    losses = []
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros(tx.shape[1])

    for iter in range(max_iter):
        loss = logistic_pen_loss(y, tx, w, lambda_) 
        grad = compute_gradient_logistic_pen(y, tx, w, lambda_)  
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
        f1, acc, val_loss = evaluate_logistic_model(y_val, X_val, w, threshold=alpha, lambda_=lambda_)
        accuracies.append(acc)
        f1_scores.append(f1)
        print("     from now, f1 score:", f1, "accuracy:", acc)       
        current += fold_size
        current_batch += 1 
        
    mean_accuracy = np.mean(accuracies)
    mean_f1 = np.mean(f1_scores)    
    return mean_accuracy, mean_f1



# Generic functions

def IdPreprocess(a, b, c, d):
    print("\nYou are not preprocessing data here\n")
    return a, b, c, d


def over_under_fitting(
        X_train, X_val, Y_train, Y_val,  
        train_method = logistic_regression_penalized_gradient_descent, 
        preprocess = IdPreprocess, 
        evaluator = evaluate_logistic_model, steps = 20):
    """
    Args:
        X_.: not preprocessed data
        Y_.: with labels 1 and 0
        train_method(Y_train, X_train):  
            return final_loss, final_w

        preprocess(X_train, X_val, Y_train, Y_val):
            return X_train_process, X_val_process, Y_train_process, Y_val_process
        evaluator(Y_val, X_val, w):
            return accuracy, f1_score, loss
        steps: number of trainings
    Returns:
        train_losses
        val_losses
    """

    train_losses = []
    val_losses = []
    f1_scores = []
    N = X_train.shape[0]

    # preprocess with the give function

    print("Before preprocess", X_train.shape, Y_train.shape)
    X_train, X_val, Y_train, Y_val = preprocess(X_train, X_val, Y_train, Y_val)
    print("After preprocess", X_train.shape, Y_train.shape)

    for i in np.arange(steps, 0, -1):
        n_i = N // i
        X_i = X_train[:n_i]
        Y_i = Y_train[:n_i]
        train_loss_i, w_i = train_method(Y_i, X_i)
        f1_i, _, val_loss_i = evaluator(Y_val, X_val, w_i)
        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)
        f1_scores.append(f1_i)
        print("\nValidation loss:", val_loss_i,"f1 score", f1_i, "with data size of", X_i.shape[0])

    return train_losses, val_losses, f1_scores





