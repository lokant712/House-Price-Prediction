"""
Retraining Script - House Price Prediction
================──────────────────────────
Allows faculty or users to easily retrain models and preprocessors for any dataset.
Saves serialized model pipelines to `outputs/models/`.

Usage:
    python retrain.py --dataset california
    python retrain.py --dataset ames
    python retrain.py --dataset uci
    python retrain.py --dataset all
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.abspath("src"))

from utils import setup_logging, ensure_dirs
from preprocessing import (
    load_california_data, load_ames_data, load_uci_data,
    get_feature_lists, get_preprocessing_pipeline
)
from models import (
    train_simple_linear_regression,
    train_multiple_linear_regression,
    train_generalized_linear_model,
    tune_decision_tree,
    tune_random_forest,
    tune_gradient_boosting,
    save_model
)

def retrain_dataset(dataset_name):
    print(f"\n==================================================")
    print(f" RETRAINING PIPELINE FOR {dataset_name.upper()}")
    print(f"==================================================")
    
    ensure_dirs(["outputs/models"])
    
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
    
    preprocessor = get_preprocessing_pipeline(num_features, cat_features)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Save Preprocessor
    preproc_path = f"outputs/models/{dataset_name}_preprocessor.joblib"
    save_model(preprocessor, preproc_path)
    print(f"[OK] Saved Preprocessor -> {preproc_path}")
    
    # Target Transformation (Ames)
    target_transform = 'log' if dataset_name == 'ames' else None
    y_train_fit = np.log1p(y_train_orig) if target_transform == 'log' else y_train_orig.copy()
    
    # Train Models
    models = {
        'simple_linear': train_simple_linear_regression(X_train_proc, y_train_fit),
        'multiple_linear': train_multiple_linear_regression(X_train_proc, y_train_fit),
        'glm': train_generalized_linear_model(X_train_proc, y_train_fit),
        'decision_tree': tune_decision_tree(X_train_proc, y_train_fit),
        'random_forest': tune_random_forest(X_train_proc, y_train_fit),
        'gradient_boosting': tune_gradient_boosting(X_train_proc, y_train_fit)
    }
    
    for name, model in models.items():
        model_path = f"outputs/models/{dataset_name}_{name}.joblib"
        save_model(model, model_path)
        print(f"[OK] Saved Model: {name} -> {model_path}")
        
    print(f"Retraining completed successfully for {dataset_name}!")

def main():
    parser = argparse.ArgumentParser(description="Retrain House Price Prediction Models")
    parser.add_argument(
        "--dataset",
        type=str,
        default="california",
        choices=["california", "ames", "uci", "all"],
        help="Dataset to retrain (california, ames, uci, or all)"
    )
    args = parser.parse_args()
    
    setup_logging("outputs/pipeline.log")
    
    if args.dataset == "all":
        for ds in ["california", "ames", "uci"]:
            retrain_dataset(ds)
    else:
        retrain_dataset(args.dataset)

if __name__ == '__main__':
    main()
