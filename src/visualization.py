import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging

# Set style for publication quality
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
sns.set_theme(style="whitegrid", palette="muted")

def plot_target_distribution(df, target_col, dataset_name, output_dir="outputs/figures"):
    """Plots and saves the target variable distribution (and its log if skewed)."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{dataset_name}_target_dist.png"
    filepath = os.path.join(output_dir, filename)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5)) if dataset_name == 'ames' else plt.subplots(1, 1, figsize=(8, 5))
    
    if dataset_name == 'ames':
        # Raw Target
        sns.histplot(df[target_col], kde=True, ax=axes[0], color="royalblue")
        axes[0].set_title(f"Original {target_col} Distribution", fontweight='bold')
        axes[0].set_xlabel(target_col)
        
        # Log Target
        sns.histplot(np.log1p(df[target_col]), kde=True, ax=axes[1], color="teal")
        axes[1].set_title(f"Log-Transformed {target_col} Distribution", fontweight='bold')
        axes[1].set_xlabel(f"log({target_col} + 1)")
    else:
        ax = axes
        sns.histplot(df[target_col], kde=True, ax=ax, color="royalblue")
        ax.set_title(f"{target_col} Distribution - {dataset_name.capitalize()}", fontweight='bold')
        ax.set_xlabel(target_col)
        
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    logging.info(f"Saved target distribution plot to {filepath}")
    return filepath

def plot_correlation_heatmap(df, num_features, dataset_name, output_dir="outputs/figures"):
    """Plots and saves a correlation heatmap for numerical features."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{dataset_name}_correlation_heatmap.png"
    filepath = os.path.join(output_dir, filename)
    
    # Calculate correlation matrix
    corr = df[num_features].corr()
    
    plt.figure(figsize=(10, 8))
    # Draw heatmap with a nice coolwarm color palette
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, 
                linewidths=0.5, cbar_kws={"shrink": 0.8}, annot_kws={"size": 8})
    plt.title(f"Correlation Heatmap - {dataset_name.capitalize()} Dataset", fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    logging.info(f"Saved correlation heatmap to {filepath}")
    return filepath

def plot_outliers_box(df, num_features, dataset_name, output_dir="outputs/figures"):
    """Plots box plots for key features to visualize outliers."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{dataset_name}_outliers_boxplot.png"
    filepath = os.path.join(output_dir, filename)
    
    # We will pick a subset of important features if there are too many (e.g. for Ames)
    features_to_plot = num_features[:8] if len(num_features) > 8 else num_features
    
    # Normalize features for visual scaling in a single boxplot
    df_norm = (df[features_to_plot] - df[features_to_plot].mean()) / df[features_to_plot].std()
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_norm, orient="h", palette="Set2")
    plt.title(f"Box Plot of Standardized Features (Outlier Analysis) - {dataset_name.capitalize()}", fontweight='bold')
    plt.xlabel("Standardized Value (Z-Score)")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    logging.info(f"Saved box plot to {filepath}")
    return filepath

def plot_feature_relationships(df, num_features, target_col, dataset_name, output_dir="outputs/figures"):
    """Plots scatter plots of the top 3 correlated features with target."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{dataset_name}_feature_relationships.png"
    filepath = os.path.join(output_dir, filename)
    
    # Find top 3 correlated features with target
    corr = df[num_features + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
    # The first element is the target itself, so take the next 3
    top_features = corr.index[1:4].tolist()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, col in enumerate(top_features):
        sns.scatterplot(data=df, x=col, y=target_col, alpha=0.5, ax=axes[i], color="purple")
        sns.regplot(data=df, x=col, y=target_col, scatter=False, ax=axes[i], color="red", line_kws={"linewidth": 2})
        axes[i].set_title(f"{target_col} vs {col}", fontweight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(target_col)
        
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    logging.info(f"Saved feature relationships scatter plots to {filepath}")
    return filepath

def plot_actual_vs_predicted(y_true, y_pred, model_name, dataset_name, output_dir="outputs/figures"):
    """Plots actual vs predicted values with a diagonal reference line."""
    os.makedirs(output_dir, exist_ok=True)
    clean_model_name = model_name.replace(" ", "_").lower()
    filename = f"{dataset_name}_{clean_model_name}_actual_vs_pred.png"
    filepath = os.path.join(output_dir, filename)
    
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, color="dodgerblue")
    
    # Diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.title(f"Actual vs. Predicted - {model_name}\n({dataset_name.capitalize()} Dataset)", fontweight='bold')
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    return filepath

def plot_residual_analysis(y_true, y_pred, model_name, dataset_name, output_dir="outputs/figures"):
    """Plots residuals vs predicted values and the residual histogram."""
    os.makedirs(output_dir, exist_ok=True)
    clean_model_name = model_name.replace(" ", "_").lower()
    filename = f"{dataset_name}_{clean_model_name}_residuals.png"
    filepath = os.path.join(output_dir, filename)
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Predicted
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, ax=axes[0], color="crimson")
    axes[0].axhline(0, color='black', linestyle='--', lw=2)
    axes[0].set_title(f"Residuals vs. Predicted Values\n({model_name})", fontweight='bold')
    axes[0].set_xlabel("Predicted Values")
    axes[0].set_ylabel("Residuals (Actual - Predicted)")
    
    # Residuals Distribution
    sns.histplot(residuals, kde=True, ax=axes[1], color="darkorange")
    axes[1].axvline(0, color='black', linestyle='--', lw=2)
    axes[1].set_title("Distribution of Residuals", fontweight='bold')
    axes[1].set_xlabel("Residual Value")
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    return filepath

def plot_feature_importance(model, feature_names, model_name, dataset_name, output_dir="outputs/figures"):
    """Plots feature importances for tree models or coefficients for linear models."""
    os.makedirs(output_dir, exist_ok=True)
    clean_model_name = model_name.replace(" ", "_").lower()
    filename = f"{dataset_name}_{clean_model_name}_importance.png"
    filepath = os.path.join(output_dir, filename)
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        title = f"Feature Importances - {model_name}"
    elif hasattr(model, 'coef_'):
        # For multiple linear regression (coef_ has length of features)
        importances = np.abs(model.coef_)
        title = f"Coefficient Magnitudes (Absolute Value) - {model_name}"
    else:
        # Simple Linear regression or other models with no coefficients
        return None
        
    # Match feature names. Note: OneHotEncoder expands categorical features, so we need to match length.
    if len(importances) != len(feature_names):
        logging.warning(f"Length of importances ({len(importances)}) does not match feature names ({len(feature_names)}). Truncating/padding.")
        if len(importances) > len(feature_names):
            importances = importances[:len(feature_names)]
        else:
            feature_names = feature_names[:len(importances)]
            
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Only plot top 15 features if there are too many (e.g. after OneHotEncoding in Ames)
    imp_df_plot = imp_df.head(15)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp_df_plot, x='Importance', y='Feature', palette="viridis")
    plt.title(f"{title}\n({dataset_name.capitalize()} Dataset)", fontweight='bold')
    plt.xlabel("Importance / Coefficient Magnitude")
    plt.ylabel("Feature Name")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    return filepath

def plot_model_comparison(metrics_df, dataset_name, output_dir="outputs/figures"):
    """Generates comparative charts (R2 and RMSE) for all models."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{dataset_name}_model_comparison.png"
    filepath = os.path.join(output_dir, filename)
    
    # Sort models by testing R2 score
    metrics_df = metrics_df.sort_values(by='R2', ascending=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # R2 Comparison
    sns.barplot(data=metrics_df, x='R2', y='Model', ax=axes[0], palette="Blues_d")
    axes[0].set_title(f"Model R² Score Comparison (Higher is Better)\n{dataset_name.capitalize()} Dataset", fontweight='bold')
    axes[0].set_xlabel("R² Score")
    axes[0].set_xlim(0, 1.0)
    
    # RMSE Comparison
    sns.barplot(data=metrics_df, x='RMSE', y='Model', ax=axes[1], palette="Oranges_d")
    axes[1].set_title(f"Model RMSE Comparison (Lower is Better)\n{dataset_name.capitalize()} Dataset", fontweight='bold')
    axes[1].set_xlabel("Root Mean Squared Error")
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    logging.info(f"Saved model comparison plot to {filepath}")
    return filepath
