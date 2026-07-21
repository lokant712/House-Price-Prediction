# House Price Prediction using Machine Learning & Regression Pipelines

## Project Overview
This repository contains the complete, end-to-end predictive analytics laboratory project implementing machine learning pipelines to predict residential house prices. The project conducts a comparative empirical evaluation of regression algorithms across three real-world datasets:
1. **California Housing Dataset** (fetched from `scikit-learn`)
2. **Ames Housing Dataset** (Iowa, USA)
3. **UCI Real Estate Valuation Dataset** (Sindian, Taiwan)

The project separates concerns into modular source scripts (`src/`), interactive notebooks (`notebooks/`), standalone evaluation/inference verification entry points (`retrain.py`, `inference.py`, `evaluate.py`, `gui.py`), and automated technical report compiling.

---

## Features
- **Reproducible ML Pipelines**: Built using `scikit-learn` `Pipeline` and `ColumnTransformer` to automate data imputation (`SimpleImputer`), standardized scaling (`StandardScaler`), and categorical encoding (`OneHotEncoder`) without data leakage.
- **Robust Outlier & Skewness Management**: Implements log-transformations (`log1p`) on right-skewed target variables and outlier filtering on spatial living areas.
- **Hyperparameter Optimization**: Automated grid-search optimization using 5-fold cross-validation (`GridSearchCV`) for tree-based estimators.
- **Multiple Algorithms Evaluated**:
  1. **Simple Linear Regression** (Single primary feature)
  2. **Multiple Linear Regression** (All numeric & categorical features)
  3. **Generalized Linear Model (GLM/GI)** (Tweedie Regressor with log-link)
  4. **Decision Tree Regressor** (Hyperparameter tuned)
  5. **Random Forest Regressor** (Bagging ensemble)
  6. **Gradient Boosting Regressor** (Sequential boosting ensemble)
- **Standalone CLI & GUI Verification**: Includes dedicated Python scripts for retraining, inference, metric evaluation, and an interactive Gradio web GUI.
- **Publication-Quality Visualisations**: Generates feature correlation heatmaps, residual distribution curves, outlier box plots, feature importances, and actual-vs-predicted scatter plots.

---

## Datasets
- **California Housing**: Block-level demographic and economic aggregates (20,640 records, 8 features).
- **Ames Housing**: Residential property characteristics (1,460 records, 80 structural and location features).
- **UCI Real Estate Valuation**: Urban density and geographic proximity transaction history (414 records, 6 features).

---

## Repository Structure
```
House-Price-Prediction/
├── datasets/
│   ├── ames/
│   │   └── train.csv
│   └── uci/
│       └── Real estate valuation data set.xlsx
├── notebooks/
│   ├── california.ipynb
│   ├── ames.ipynb
│   └── uci.ipynb
├── src/
│   ├── preprocessing.py      # Loaders, imputation, scaling, column transformers
│   ├── models.py             # Model training, hyperparameter grid search, save/load
│   ├── evaluation.py         # MAE, MSE, RMSE, R2, Adjusted R2, CV scores
│   ├── visualization.py      # Heatmaps, residuals, feature importances
│   ├── utils.py              # Directory creation, logging, JSON/CSV tools
│   ├── run_pipeline.py       # Master pipeline orchestrator per dataset
│   └── generate_report.py    # Automated report compiler (.docx / .pdf)
├── outputs/                  # Auto-generated outputs (Git-ignored large binaries)
│   ├── figures/              # Publication quality plots (.png)
│   ├── tables/               # Formatted CSV and Markdown performance tables
│   ├── metrics/              # Structured JSON metric summaries
│   └── models/               # Serialized model pipelines (.joblib)
├── retrain.py                # Standalone model retraining script
├── inference.py              # Command-line price prediction script
├── evaluate.py               # Command-line metric evaluation script
├── gui.py                    # Interactive 2-line Gradio Web GUI script
├── run_all.py                # Master pipeline and report runner
├── requirements.txt          # Python dependency specifications
├── README.md                 # Project documentation & setup instructions
└── LICENSE                   # MIT License
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/lokant712/House-Price-Prediction.git
cd House-Price-Prediction
```

### 2. Set up Virtual Environment (Recommended)
**On Windows (PowerShell / Command Prompt):**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run the Project & Verification Scripts

### 1. Run Master Pipeline (End-to-End Execution)
To run data preprocessing, train all models, generate all figures, export tables, compute metrics, and build technical reports:
```bash
python run_all.py
```

### 2. Retrain Models (`retrain.py`)
To retrain preprocessors and regression models and export model pipelines (`.joblib`):
```bash
# Retrain California dataset models
python retrain.py --dataset california

# Retrain all datasets (California, Ames, UCI)
python retrain.py --dataset all
```

### 3. Evaluate Models (`evaluate.py`)
To evaluate saved models against test splits and display MAE, RMSE, $R^2$, Adjusted $R^2$, and 5-fold CV scores:
```bash
python evaluate.py --dataset california
python evaluate.py --dataset ames
python evaluate.py --dataset uci
```

### 4. Run Model Inference (`inference.py`)
To perform price predictions on sample data using saved models:
```bash
python inference.py --dataset california --model gradient_boosting
python inference.py --dataset ames --model random_forest
```

### 5. Launch Interactive Web GUI (`gui.py`)
To start an interactive Gradio web interface in your browser for live price predictions:
```bash
python gui.py
```
*Navigating to `http://127.0.0.1:7860` opens an interactive dashboard with sliders for property features.*

### 6. Interactive Jupyter Notebooks
Launch Jupyter to explore interactive notebooks:
```bash
jupyter notebook notebooks/california.ipynb
```

---

## Generated Project Outputs

All project outputs are saved automatically in the `outputs/` directory:

1. **`outputs/figures/`**:
   - `target_distribution_<dataset>.png`: Density and skewness curves.
   - `correlation_heatmap_<dataset>.png`: Feature correlation matrices.
   - `residual_analysis_<dataset>_<model>.png`: Residual distribution & residual-vs-fitted scatter plots.
   - `feature_importance_<dataset>_<model>.png`: Bar charts of top feature weights.
   - `model_comparison_<dataset>.png`: Bar charts comparing $R^2$ scores across all models.

2. **`outputs/tables/`**:
   - `model_comparison_<dataset>.csv`: Metric summary per dataset.
   - `cross_dataset_comparison.md`: Global comparison matrix.

3. **`outputs/metrics/`**:
   - `<dataset>_metrics.json`: JSON payload containing exact numerical evaluation metrics.

4. **`outputs/models/`**:
   - `<dataset>_preprocessor.joblib`: Serialized ColumnTransformer.
   - `<dataset>_<model>.joblib`: Serialized fitted model objects.

---

## Summary of Results

| Dataset | Best Model | Test R² | Test RMSE | Primary Features |
| :--- | :--- | :---: | :---: | :--- |
| **California** | Gradient Boosting Regressor | `0.7932` | `0.5262` | Median Income (`MedInc`), Latitude, Longitude |
| **Ames** | Gradient Boosting Regressor | `0.8984` | `25,007` | Overall Quality, GrLivArea, Garage Cars |
| **UCI Real Estate** | Random Forest Regressor | `0.7352` | `6.7118` | Distance to MRT (`X3`), House Age |

---

## License
Distributed under the MIT License. See `LICENSE` for details.
