import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Import project modules
from utils import setup_logging, ensure_dirs, save_json, save_table
from preprocessing import (
    load_california_data, load_ames_data, load_uci_data,
    get_feature_lists, get_preprocessing_pipeline
)
from models import (
    train_simple_linear_regression, train_multiple_linear_regression,
    train_generalized_linear_model, tune_decision_tree,
    tune_random_forest, tune_gradient_boosting, save_model
)
from evaluation import evaluate_model
from visualization import (
    plot_target_distribution, plot_correlation_heatmap, plot_outliers_box,
    plot_feature_relationships, plot_actual_vs_predicted, plot_residual_analysis,
    plot_feature_importance, plot_model_comparison
)

def run_dataset_pipeline(dataset_name):
    logging.info(f"\n==================================================")
    logging.info(f"STARTING PIPELINE FOR {dataset_name.upper()} DATASET")
    logging.info(f"==================================================")
    
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
        raise ValueError(f"Unknown dataset name: {dataset_name}")
        
    # 2. Get Features Lists and ensure primary_feature is the first numerical feature
    num_features, cat_features = get_feature_lists(df, target_col, dataset_name)
    
    # Put primary feature first to ensure it's at index 0 in the preprocessed matrix
    if primary_feature in num_features:
        num_features.remove(primary_feature)
    num_features = [primary_feature] + num_features
    
    logging.info(f"Numerical features (Primary: {primary_feature}): {len(num_features)}")
    logging.info(f"Categorical features: {len(cat_features)}")
    
    # 3. Exploratory Data Analysis (EDA) Visualizations
    plot_target_distribution(df, target_col, dataset_name)
    plot_correlation_heatmap(df, num_features, dataset_name)
    plot_outliers_box(df, num_features, dataset_name)
    plot_feature_relationships(df, num_features, target_col, dataset_name)
    
    # 4. Train-Test Split (80/20)
    X = df.drop(columns=[target_col])
    if 'Id' in X.columns and dataset_name == 'ames':
        X = X.drop(columns=['Id'])
    y = df[target_col]
    
    X_train, X_test, y_train_orig, y_test_orig = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logging.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # 5. Get Preprocessing Pipeline & Fit
    preprocessor = get_preprocessing_pipeline(num_features, cat_features)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing (especially for OneHotEncoder)
    # This is useful for feature importance plot
    if len(cat_features) > 0:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = list(cat_encoder.get_feature_names_out(cat_features))
        feature_names = num_features + cat_feature_names
    else:
        feature_names = num_features
        
    # 6. Target Transformation (Ames only)
    # Log transform right-skewed Target for Ames
    target_transform = 'log' if dataset_name == 'ames' else None
    if target_transform == 'log':
        y_train_fit = np.log1p(y_train_orig)
    else:
        y_train_fit = y_train_orig.copy()
        
    # Save Preprocessor Artifact
    ensure_dirs(["outputs/models"])
    save_model(preprocessor, f"outputs/models/{dataset_name}_preprocessor.joblib")
    
    # 7. Model Training & Tuning
    models_dict = {}
    
    # Model 1: Simple Linear Regression
    slr = train_simple_linear_regression(X_train_proc, y_train_fit)
    models_dict['Simple Linear Regression'] = slr
    
    # Model 2: Multiple Linear Regression
    mlr = train_multiple_linear_regression(X_train_proc, y_train_fit)
    models_dict['Multiple Linear Regression'] = mlr
    
    # Model 3: Generalized Linear Model (GLM/GI)
    glm = train_generalized_linear_model(X_train_proc, y_train_fit)
    models_dict['Generalized Linear Model (GLM)'] = glm
    
    # Model 4: Decision Tree
    dt = tune_decision_tree(X_train_proc, y_train_fit, cv=5)
    models_dict['Decision Tree Regressor'] = dt
    
    # Model 5: Random Forest
    rf = tune_random_forest(X_train_proc, y_train_fit, cv=5)
    models_dict['Random Forest Regressor'] = rf
    
    # Model 6: Gradient Boosting
    gb = tune_gradient_boosting(X_train_proc, y_train_fit, cv=5)
    models_dict['Gradient Boosting Regressor'] = gb

    # Save trained model artifacts
    key_mapping = {
        'Simple Linear Regression': 'simple_linear',
        'Multiple Linear Regression': 'multiple_linear',
        'Generalized Linear Model (GLM)': 'glm',
        'Decision Tree Regressor': 'decision_tree',
        'Random Forest Regressor': 'random_forest',
        'Gradient Boosting Regressor': 'gradient_boosting'
    }
    for m_name, m_obj in models_dict.items():
        save_model(m_obj, f"outputs/models/{dataset_name}_{key_mapping[m_name]}.joblib")

    
    # 8. Evaluation and Plotting
    metrics_list = []
    
    for name, model in models_dict.items():
        is_simple = (name == 'Simple Linear Regression')
        
        # Evaluate model (metrics on original scale)
        metrics, y_pred_tr, y_pred_te = evaluate_model(
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
        
        # Save evaluation plots
        plot_actual_vs_predicted(y_test_orig, y_pred_te, name, dataset_name)
        plot_residual_analysis(y_test_orig, y_pred_te, name, dataset_name)
        
        # Plot feature importance / coefficients if applicable
        if name == 'Multiple Linear Regression':
            plot_feature_importance(model, feature_names, name, dataset_name)
        elif name in ['Decision Tree Regressor', 'Random Forest Regressor', 'Gradient Boosting Regressor']:
            plot_feature_importance(model, feature_names, name, dataset_name)
            
        metrics['Model'] = name
        metrics_list.append(metrics)
        
        logging.info(f"Model: {name:<28} | Test R2: {metrics['Testing_Score']:.4f} | Test RMSE: {metrics['RMSE']:.4f}")
        
    # 9. Model Comparison & Ranking
    metrics_df = pd.DataFrame(metrics_list)
    # Rearrange columns
    cols_order = ['Model', 'MAE', 'MSE', 'RMSE', 'R2', 'Adjusted_R2', 'CV_Score', 'Training_Score', 'Testing_Score']
    metrics_df = metrics_df[cols_order]
    
    # Rank models by Testing Score (R2)
    metrics_df['Rank'] = metrics_df['Testing_Score'].rank(ascending=False).astype(int)
    metrics_df = metrics_df.sort_values(by='Rank')
    
    # Save Metrics & Tables
    save_table(metrics_df, f"outputs/tables/{dataset_name}_metrics.csv", f"outputs/tables/{dataset_name}_metrics.md")
    save_json(metrics_df.to_dict(orient='records'), f"outputs/metrics/{dataset_name}_metrics.json")
    
    # Plot Model Comparison charts
    plot_model_comparison(metrics_df, dataset_name)
    
    # Identify Best Model
    best_model_row = metrics_df.iloc[0]
    logging.info(f"BEST MODEL FOR {dataset_name.upper()}: {best_model_row['Model']} (Test R2: {best_model_row['Testing_Score']:.4f})")
    
    return metrics_df

def run_all_pipelines():
    ensure_dirs()
    setup_logging("outputs/pipeline.log")
    
    logging.info("Starting House Price Prediction Pipelines for California, Ames, and UCI datasets...")
    
    results = {}
    for ds in ['california', 'ames', 'uci']:
        results[ds] = run_dataset_pipeline(ds)
        
    # Create cross-dataset comparison table
    logging.info("\nCreating Cross-Dataset Comparison Table...")
    cross_list = []
    for ds, df in results.items():
        for _, row in df.iterrows():
            d = row.to_dict()
            d['Dataset'] = ds.capitalize()
            cross_list.append(d)
            
    cross_df = pd.DataFrame(cross_list)
    # Order columns
    cols = ['Dataset', 'Model', 'MAE', 'MSE', 'RMSE', 'R2', 'Adjusted_R2', 'CV_Score', 'Training_Score', 'Testing_Score', 'Rank']
    cross_df = cross_df[cols]
    
    save_table(cross_df, "outputs/tables/cross_dataset_comparison.csv", "outputs/tables/cross_dataset_comparison.md")
    save_json(cross_df.to_dict(orient='records'), "outputs/metrics/cross_dataset_comparison.json")
    logging.info("Pipeline run completed successfully.")

if __name__ == '__main__':
    run_all_pipelines()
