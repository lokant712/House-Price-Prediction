import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

def load_california_data():
    """Loads the California Housing Dataset from sklearn."""
    logging.info("Fetching California Housing Dataset from sklearn...")
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    
    # Rename target to make it descriptive
    target_col = 'MedHouseVal'
    logging.info(f"California Housing data loaded. Shape: {df.shape}")
    return df, target_col

def load_ames_data(filepath='datasets/ames/train.csv'):
    """Loads the Ames Housing Dataset from the specified path."""
    logging.info(f"Loading Ames Housing Dataset from {filepath}...")
    df = pd.read_csv(filepath)
    target_col = 'SalePrice'
    
    # Simple outlier removal as per standard data science practice on Ames
    # (GrLivArea > 4000 is recommended by the author of the dataset to remove outliers)
    if 'GrLivArea' in df.columns:
        initial_len = len(df)
        df = df[df['GrLivArea'] < 4000].copy()
        logging.info(f"Removed {initial_len - len(df)} outliers (GrLivArea >= 4000)")
        
    logging.info(f"Ames Housing data loaded. Shape: {df.shape}")
    return df, target_col

def load_uci_data(filepath='datasets/uci/Real estate valuation data set.xlsx'):
    """Loads the UCI Real Estate Valuation Dataset from the specified path."""
    logging.info(f"Loading UCI Real Estate Valuation Dataset from {filepath}...")
    df = pd.read_excel(filepath)
    target_col = 'Y house price of unit area'
    
    # Drop the index column
    if 'No' in df.columns:
        df = df.drop(columns=['No'])
        
    logging.info(f"UCI Real Estate data loaded. Shape: {df.shape}")
    return df, target_col

def get_feature_lists(df, target_col, dataset_name):
    """Returns lists of numerical and categorical columns for the dataset."""
    features = [col for col in df.columns if col != target_col]
    
    if dataset_name == 'california':
        # California housing is entirely numeric
        num_features = features
        cat_features = []
    elif dataset_name == 'uci':
        # UCI real estate is entirely numeric
        num_features = features
        cat_features = []
    else: # ames
        # For Ames, we separate object columns and numeric columns
        # Exclude ID column if it exists
        if 'Id' in features:
            features.remove('Id')
        num_features = [col for col in features if df[col].dtype in ['int64', 'float64']]
        cat_features = [col for col in features if df[col].dtype == 'object']
        
    return num_features, cat_features

def get_preprocessing_pipeline(num_features, cat_features):
    """Creates a ColumnTransformer pipeline for numerical and categorical features."""
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    try:
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
    except TypeError:
        # Fallback for scikit-learn < 1.2 which uses `sparse` instead of `sparse_output`
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_features),
            ('cat', cat_pipeline, cat_features)
        ],
        remainder='drop'
    )
    
    return preprocessor
