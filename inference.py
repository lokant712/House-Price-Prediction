"""
Inference Script - House Price Prediction
================─────────────────────────
Allows faculty or users to load a trained model pipeline and run price predictions on sample data.

Usage:
    python inference.py --dataset california --model gradient_boosting
    python inference.py --dataset ames --model random_forest
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath("src"))

from preprocessing import load_california_data, load_ames_data, load_uci_data
from models import load_model

def run_inference(dataset_name, model_key):
    print(f"\n==================================================")
    print(f" RUNNING INFERENCE: {dataset_name.upper()} ({model_key})")
    print(f"==================================================")
    
    preproc_path = f"outputs/models/{dataset_name}_preprocessor.joblib"
    model_path = f"outputs/models/{dataset_name}_{model_key}.joblib"
    
    if not os.path.exists(preproc_path) or not os.path.exists(model_path):
        print(f"[!] Model artifacts missing ({preproc_path} or {model_path}). Please run `python retrain.py --dataset {dataset_name}` first.")
        return
        
    preprocessor = load_model(preproc_path)
    model = load_model(model_path)
    
    # Load sample input row from dataset
    if dataset_name == 'california':
        df, target_col = load_california_data()
    elif dataset_name == 'ames':
        df, target_col = load_ames_data()
    else: # uci
        df, target_col = load_uci_data()
        
    sample_inputs = df.drop(columns=[target_col]).head(3)
    actual_values = df[target_col].head(3).values
    
    if 'Id' in sample_inputs.columns and dataset_name == 'ames':
        sample_inputs = sample_inputs.drop(columns=['Id'])
        
    # Preprocess
    proc_inputs = preprocessor.transform(sample_inputs)
    
    # If simple linear regression, slice feature 0
    if model_key == 'simple_linear':
        proc_inputs = proc_inputs[:, 0].reshape(-1, 1)
        
    # Predict
    preds_fit = model.predict(proc_inputs)
    
    # Inverse transform target for Ames if target was log-transformed
    if dataset_name == 'ames':
        predictions = np.expm1(preds_fit)
    else:
        predictions = preds_fit
        
    print("\n--- SAMPLE INFERENCE RESULTS ---")
    for i in range(len(sample_inputs)):
        print(f"Sample #{i+1}: Predicted Price = {predictions[i]:,.2f} | Actual Price = {actual_values[i]:,.2f}")

def main():
    parser = argparse.ArgumentParser(description="Run House Price Prediction Inference")
    parser.add_argument(
        "--dataset",
        type=str,
        default="california",
        choices=["california", "ames", "uci"],
        help="Dataset name"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gradient_boosting",
        choices=["simple_linear", "multiple_linear", "glm", "decision_tree", "random_forest", "gradient_boosting"],
        help="Model architecture"
    )
    args = parser.parse_args()
    
    run_inference(args.dataset, args.model)

if __name__ == '__main__':
    main()
