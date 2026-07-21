"""
Evaluation Script - House Price Prediction
================─────────────────────────
Allows faculty or users to evaluate trained regression models on test data.

Usage:
    python evaluate.py --dataset california
    python evaluate.py --dataset ames
    python evaluate.py --dataset uci
    python evaluate.py --dataset all
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath("src"))

from utils import setup_logging
from preprocessing import (
    load_california_data, load_ames_data, load_uci_data,
    get_feature_lists
)
from models import load_model
from evaluation import evaluate_model

def evaluate_dataset(dataset_name):
    print(f"\n==================================================")
    print(f" EVALUATING MODELS FOR {dataset_name.upper()}")
    print(f"==================================================")
    
    # 1. Load Data
    if dataset_name == 'california':
        df, target_col = load_california_data()
        primary_feature = 'MedInc'
    elif dataset_name == 'ames':
        df, target_col = load_ames_data()
        primary_feature = 'GrLivArea'
    elif dataset_name == 'uci':
        df, target_col = load_uci_data()
        primary_feature = 'X3 distance to the nearest MRT station'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    num_features, cat_features = get_feature_lists(df, target_col, dataset_name)
    if primary_feature in num_features:
        num_features.remove(primary_feature)
    num_features = [primary_feature] + num_features
    
    X = df.drop(columns=[target_col])
    if 'Id' in X.columns and dataset_name == 'ames':
        X = X.drop(columns=['Id'])
    y = df[target_col]
    
    X_train, X_test, y_train_orig, y_test_orig = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    preproc_path = f"outputs/models/{dataset_name}_preprocessor.joblib"
    if not os.path.exists(preproc_path):
        print(f"[!] Preprocessor not found at {preproc_path}. Please run retrain.py first.")
        return
        
    preprocessor = load_model(preproc_path)
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    target_transform = 'log' if dataset_name == 'ames' else None
    y_train_fit = np.log1p(y_train_orig) if target_transform == 'log' else y_train_orig.copy()
    
    model_names = [
        ('Simple Linear Regression', 'simple_linear', True),
        ('Multiple Linear Regression', 'multiple_linear', False),
        ('Generalized Linear Model (GLM)', 'glm', False),
        ('Decision Tree Regressor', 'decision_tree', False),
        ('Random Forest Regressor', 'random_forest', False),
        ('Gradient Boosting Regressor', 'gradient_boosting', False)
    ]
    
    results_list = []
    
    for display_name, file_key, is_simple in model_names:
        model_path = f"outputs/models/{dataset_name}_{file_key}.joblib"
        if not os.path.exists(model_path):
            print(f"[!] Warning: Model file {model_path} missing. Skipping.")
            continue
            
        model = load_model(model_path)
        metrics, _, _ = evaluate_model(
            model=model,
            X_train=X_train_proc,
            y_train_fit=y_train_fit,
            X_test=X_test_proc,
            y_test_orig=y_test_orig,
            y_train_orig=y_train_orig,
            is_simple=is_simple,
            target_transform=target_transform,
            cv=5
        )
        
        metrics['Model'] = display_name
        results_list.append(metrics)
        
    if results_list:
        results_df = pd.DataFrame(results_list)
        cols = ['Model', 'MAE', 'RMSE', 'R2', 'Adjusted_R2', 'CV_Score']
        results_df = results_df[cols]
        print("\n" + results_df.to_string(index=False))
    else:
        print("No models evaluated.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate House Price Prediction Models")
    parser.add_argument(
        "--dataset",
        type=str,
        default="california",
        choices=["california", "ames", "uci", "all"],
        help="Dataset to evaluate (california, ames, uci, or all)"
    )
    args = parser.parse_args()
    
    setup_logging("outputs/pipeline.log")
    
    if args.dataset == "all":
        for ds in ["california", "ames", "uci"]:
            evaluate_dataset(ds)
    else:
        evaluate_dataset(args.dataset)

if __name__ == '__main__':
    main()
