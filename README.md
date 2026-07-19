# House Price Prediction using Regression Models

## Project Overview
This repository contains the complete, end-to-end laboratory project implementing machine learning pipelines to predict residential house prices. The project conducts a comparative empirical evaluation of five distinct regression algorithms across three diverse real-world datasets:
1. **California Housing Dataset** (fetched from scikit-learn)
2. **Ames Housing Dataset** (Iowa, USA)
3. **UCI Real Estate Valuation Dataset** (Sindian, Taiwan)

The project structure separates concerns into structured source modules (`src/`), interactive notebooks (`notebooks/`), and automated test scripts. It concludes with the generation of professional technical reports in Word and PDF formats containing in-depth analyses of residual distributions, model assumptions, and feature importances.

## Features
- **Reproducible Pipelines**: Built using scikit-learn `Pipeline` and `ColumnTransformer` to automate data imputation, standardized scaling, and high-dimensional categorical encoding without data leakage.
- **Robust Outlier & Skewness Management**: Implements log-transformations (`log1p`) on right-skewed target variables and outlier filtering on spatial living areas to satisfy linear model assumptions.
- **Hyperparameter Optimization**: Automated grid-search optimization using 5-fold cross-validation (`GridSearchCV`) for tree-based estimators.
- **Publication-Quality Visualisations**: Generates feature correlation heatmaps, residual distribution curves, outlier box plots, feature importances, and actual-vs-predicted scatter plots.
- **Automated Reporting**: A Python compiler script programmatically builds a complete university project report in `.docx` and `.pdf` formats, embedding all tables and figures dynamically.

## Datasets
- **California Housing**: Block-level demographic and economic aggregates (20,640 records, 8 features).
- **Ames Housing**: Residential property characteristics (1,460 records, 80 structural and location features).
- **UCI Real Estate Valuation**: Urban density and geographic proximity transaction history (414 records, 6 features).

## Models
Only the following five algorithms are implemented, compared, and ranked:
1. **Simple Linear Regression**: OLS fit using the single most correlated feature for each dataset.
2. **Multiple Linear Regression**: OLS fit using all preprocessed numerical and categorical features.
3. **Decision Tree Regressor**: Optimized tree partitioning.
4. **Random Forest Regressor**: Bagging tree ensemble.
5. **Gradient Boosting Regressor**: Sequential boosting ensemble.

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
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── visualization.py
│   ├── utils.py
│   └── generate_report.py
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── metrics/
├── reports/
│   ├── report.docx
│   └── report.pdf
├── requirements.txt
├── README.md
├── LICENSE
└── run_all.py
```

---

## Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/lokant712/House-Price-Prediction.git
   cd House-Price-Prediction
   ```

2. **Set up virtual environment (Recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

*Note: Microsoft Word is required to compile and convert `reports/report.docx` to `reports/report.pdf` using the automated runner.*

---

## Usage

### Run Entire Pipeline (Automated)
To run data preprocessing, train all models, generate all tables, figures, metrics, compile the Word report, and export the PDF report:
```bash
python run_all.py
```

### Run Notebooks Interactively
Launch Jupyter notebook and explore individual analysis files under `notebooks/`:
```bash
jupyter notebook notebooks/california.ipynb
```

---

## Summary of Results

### Cross-Dataset Best Model Performance

| Dataset | Best Model | Test R² | Test RMSE | Primary Features |
| :--- | :--- | :---: | :---: | :--- |
| **California** | Gradient Boosting Regressor | `0.7932` | `0.5262` | Median Income (`MedInc`), Latitude, Longitude |
| **Ames** | Gradient Boosting Regressor | `0.8984` | `25,007` | Overall Quality, GrLivArea, Garage Cars |
| **UCI Real Estate** | Random Forest Regressor | `0.7352` | `6.7118` | Distance to MRT (`X3`), House Age |

*For complete comparative tables and metrics, refer to `outputs/tables/cross_dataset_comparison.md`.*

---

## Future Work
- Implement geospatial modeling (e.g., Kriging or Geographically Weighted Regression) to improve coordination analysis.
- Integrate time-series economic variables (inflation, interest rates) to make predictions dynamic.
- Utilize deep learning architectures (e.g., TabNet or Multi-layer Perceptrons) for high-cardinality nominal category learning.

## License
Distributed under the MIT License. See `LICENSE` for details.
