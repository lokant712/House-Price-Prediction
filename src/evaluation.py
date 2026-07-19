import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import logging

def calculate_metrics(y_true, y_pred, n_samples, n_features):
    """Calculates regression metrics: MAE, MSE, RMSE, R2, and Adjusted R2."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Adjusted R2
    denom = n_samples - n_features - 1
    if denom > 0:
        adj_r2 = 1 - (1 - r2) * (n_samples - 1) / denom
    else:
        adj_r2 = r2  # Fallback if too many features
        
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Adjusted_R2': adj_r2
    }

def evaluate_model(model, X_train, y_train_fit, X_test, y_test_orig, y_train_orig, 
                   is_simple=False, target_transform=None, cv=5):
    """
    Evaluates a model and returns a dictionary of metrics.
    
    Parameters:
      model: The trained model object.
      X_train, X_test: Preprocessed feature matrices.
      y_train_fit: The target variable used during fitting (could be log-transformed).
      y_test_orig: Original scale target for test set.
      y_train_orig: Original scale target for train set.
      is_simple: If True, model is simple linear regression (uses only first column).
      target_transform: 'log' or None. If 'log', target was log1p-transformed.
      cv: Number of cross-validation folds.
    """
    # Slice features for Simple Linear Regression
    X_tr = X_train[:, 0].reshape(-1, 1) if is_simple else X_train
    X_te = X_test[:, 0].reshape(-1, 1) if is_simple else X_test
    
    # Number of features
    n_features = 1 if is_simple else X_train.shape[1]
    
    # Predictions (on fit scale)
    y_pred_tr_fit = model.predict(X_tr)
    y_pred_te_fit = model.predict(X_te)
    
    # Convert predictions back to original scale if needed
    if target_transform == 'log':
        y_pred_tr = np.expm1(y_pred_tr_fit)
        y_pred_te = np.expm1(y_pred_te_fit)
    else:
        y_pred_tr = y_pred_tr_fit
        y_pred_te = y_pred_te_fit
        
    # Calculate metrics on original scale
    train_metrics = calculate_metrics(y_train_orig, y_pred_tr, len(y_train_orig), n_features)
    test_metrics = calculate_metrics(y_test_orig, y_pred_te, len(y_test_orig), n_features)
    
    # Cross validation score (run on the fit scale, as R2)
    # Using the appropriate slice of X_train
    cv_scores = cross_val_score(model, X_tr, y_train_fit, cv=cv, scoring='r2')
    cv_mean = np.mean(cv_scores)
    
    # Return consolidated metrics
    results = {
        'MAE': test_metrics['MAE'],
        'MSE': test_metrics['MSE'],
        'RMSE': test_metrics['RMSE'],
        'R2': test_metrics['R2'],
        'Adjusted_R2': test_metrics['Adjusted_R2'],
        'CV_Score': cv_mean,
        'Training_Score': train_metrics['R2'],
        'Testing_Score': test_metrics['R2']
    }
    return results, y_pred_tr, y_pred_te
