from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import logging

def train_simple_linear_regression(X_train, y_train):
    """Trains a Simple Linear Regression model using only the first feature in the matrix."""
    logging.info("Training Simple Linear Regression (using single feature)...")
    # Slice the first column only. Reshape to ensure 2D array for sklearn.
    X_train_single = X_train[:, 0].reshape(-1, 1)
    model = LinearRegression()
    model.fit(X_train_single, y_train)
    return model

def train_multiple_linear_regression(X_train, y_train):
    """Trains a Multiple Linear Regression model using all preprocessed features."""
    logging.info("Training Multiple Linear Regression (using all features)...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def tune_decision_tree(X_train, y_train, cv=5):
    """Tunes hyperparameters for DecisionTreeRegressor using GridSearchCV."""
    logging.info("Tuning Decision Tree Regressor...")
    model = DecisionTreeRegressor(random_state=42)
    param_grid = {
        'max_depth': [6, 10, None],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4]
    }
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='r2',
        cv=cv,
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    logging.info(f"Decision Tree Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def tune_random_forest(X_train, y_train, cv=5):
    """Tunes hyperparameters for RandomForestRegressor using GridSearchCV."""
    logging.info("Tuning Random Forest Regressor...")
    model = RandomForestRegressor(random_state=42)
    # Keeping grid sizes reasonable for execution speed while maintaining depth
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, None],
        'min_samples_split': [5, 10]
    }
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='r2',
        cv=cv,
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    logging.info(f"Random Forest Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def tune_gradient_boosting(X_train, y_train, cv=5):
    """Tunes hyperparameters for GradientBoostingRegressor using GridSearchCV."""
    logging.info("Tuning Gradient Boosting Regressor...")
    model = GradientBoostingRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='r2',
        cv=cv,
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    logging.info(f"Gradient Boosting Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_
