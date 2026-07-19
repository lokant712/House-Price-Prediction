import os
import json
import logging

def create_california_notebook():
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# California Housing Dataset - House Price Prediction\n",
                    "**Subject:** Advanced Predictive Analytics (MDI3003)  \n",
                    "**Student:** Lokanth S (23MID0037)  \n",
                    "**Faculty:** Dr. Durgesh Kumar  \n",
                    "\n",
                    "This notebook runs the complete machine learning pipeline for the **California Housing Dataset** (fetched from `sklearn`).  \n",
                    "Models implemented and compared:\n",
                    "1. Simple Linear Regression (using `MedInc` as the single predictor)\n",
                    "2. Multiple Linear Regression\n",
                    "3. Decision Tree Regressor (Hyperparameter-tuned)\n",
                    "4. Random Forest Regressor (Hyperparameter-tuned)\n",
                    "5. Gradient Boosting Regressor (Hyperparameter-tuned)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys\n",
                    "import os\n",
                    "# Append src directory to path\n",
                    "sys.path.append(os.path.abspath('../src'))\n",
                    "\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "from preprocessing import load_california_data, get_feature_lists, get_preprocessing_pipeline\n",
                    "from run_pipeline import run_dataset_pipeline\n",
                    "from utils import setup_logging\n",
                    "\n",
                    "setup_logging('../outputs/california_pipeline.log')\n",
                    "%matplotlib inline"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Data Loading & Inspection"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df, target_col = load_california_data()\n",
                    "print(f\"Target Column: {target_col}\")\n",
                    "print(f\"Shape: {df.shape}\")\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df.info()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df.describe()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Missing Value & Duplicate Analysis"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(\"Missing Values:\")\n",
                    "print(df.isnull().sum())\n",
                    "print(f\"\\nDuplicates: {df.duplicated().sum()}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Exploratory Data Analysis (EDA)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Target Variable Distribution\n",
                    "plt.figure(figsize=(8, 5))\n",
                    "sns.histplot(df[target_col], kde=True, color='royalblue')\n",
                    "plt.title(f\"Distribution of {target_col}\")\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Correlation Heatmap\n",
                    "num_features, _ = get_feature_lists(df, target_col, 'california')\n",
                    "plt.figure(figsize=(10, 8))\n",
                    "sns.heatmap(df[num_features + [target_col]].corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')\n",
                    "plt.title(\"Correlation Matrix\")\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Execute Full Pipeline\n",
                    "We will run the orchestrated pipeline that handles preprocessing, splits the data, trains and tunes the 5 models, saves results, and generates publication-quality plots."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "metrics_df = run_dataset_pipeline('california')\n",
                    "metrics_df"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open("notebooks/california.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2)
    logging.info("Created notebooks/california.ipynb")

def create_ames_notebook():
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Ames Housing Dataset - House Price Prediction\n",
                    "**Subject:** Advanced Predictive Analytics (MDI3003)  \n",
                    "**Student:** Lokanth S (23MID0037)  \n",
                    "**Faculty:** Dr. Durgesh Kumar  \n",
                    "\n",
                    "This notebook runs the complete machine learning pipeline for the **Ames Housing Dataset** (loaded from `datasets/ames/train.csv`).  \n",
                    "This dataset has 81 features and contains many missing values and categorical properties.  \n",
                    "Models compared:\n",
                    "1. Simple Linear Regression (using `GrLivArea` as predictor)\n",
                    "2. Multiple Linear Regression\n",
                    "3. Decision Tree Regressor\n",
                    "4. Random Forest Regressor\n",
                    "5. Gradient Boosting Regressor"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys\n",
                    "import os\n",
                    "# Append src directory to path\n",
                    "sys.path.append(os.path.abspath('../src'))\n",
                    "\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "from preprocessing import load_ames_data, get_feature_lists, get_preprocessing_pipeline\n",
                    "from run_pipeline import run_dataset_pipeline\n",
                    "from utils import setup_logging\n",
                    "\n",
                    "setup_logging('../outputs/ames_pipeline.log')\n",
                    "%matplotlib inline"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Data Loading & Inspection"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df, target_col = load_ames_data('../datasets/ames/train.csv')\n",
                    "print(f\"Target Column: {target_col}\")\n",
                    "print(f\"Shape: {df.shape}\")\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df.info()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Missing Value & Duplicate Analysis"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "missing = df.isnull().sum()\n",
                    "missing = missing[missing > 0].sort_values(ascending=False)\n",
                    "print(\"Columns with Missing Values:\")\n",
                    "print(missing)\n",
                    "print(f\"\\nDuplicates: {df.duplicated().sum()}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Target Distribution Analysis (Log Transform)\n",
                    "Due to right skewness, we analyze both raw and log-transformed `SalePrice`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                    "sns.histplot(df[target_col], kde=True, ax=axes[0], color='royalblue')\n",
                    "axes[0].set_title(\"Original SalePrice Distribution\")\n",
                    "\n",
                    "sns.histplot(np.log1p(df[target_col]), kde=True, ax=axes[1], color='teal')\n",
                    "axes[1].set_title(\"Log-Transformed SalePrice Distribution\")\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Run Pipeline\n",
                    "This runs categorical encoding (One-Hot), imputation, scaling, splits, and evaluates all models on the original dollar scale."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "metrics_df = run_dataset_pipeline('ames')\n",
                    "metrics_df"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open("notebooks/ames.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2)
    logging.info("Created notebooks/ames.ipynb")

def create_uci_notebook():
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# UCI Real Estate Valuation Dataset - Price Prediction\n",
                    "**Subject:** Advanced Predictive Analytics (MDI3003)  \n",
                    "**Student:** Lokanth S (23MID0037)  \n",
                    "**Faculty:** Dr. Durgesh Kumar  \n",
                    "\n",
                    "This notebook runs the complete machine learning pipeline for the **UCI Real Estate Valuation Dataset** (loaded from `datasets/uci/Real estate valuation data set.xlsx`).  \n",
                    "Models compared:\n",
                    "1. Simple Linear Regression (using `X3 distance to the nearest MRT station` as predictor)\n",
                    "2. Multiple Linear Regression\n",
                    "3. Decision Tree Regressor\n",
                    "4. Random Forest Regressor\n",
                    "5. Gradient Boosting Regressor"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys\n",
                    "import os\n",
                    "# Append src directory to path\n",
                    "sys.path.append(os.path.abspath('../src'))\n",
                    "\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "from preprocessing import load_uci_data, get_feature_lists, get_preprocessing_pipeline\n",
                    "from run_pipeline import run_dataset_pipeline\n",
                    "from utils import setup_logging\n",
                    "\n",
                    "setup_logging('../outputs/uci_pipeline.log')\n",
                    "%matplotlib inline"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Data Loading & Inspection"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df, target_col = load_uci_data('../datasets/uci/Real estate valuation data set.xlsx')\n",
                    "print(f\"Target Column: {target_col}\")\n",
                    "print(f\"Shape: {df.shape}\")\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df.info()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df.describe()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Exploratory Data Analysis"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Target Variable Distribution\n",
                    "plt.figure(figsize=(8, 5))\n",
                    "sns.histplot(df[target_col], kde=True, color='royalblue')\n",
                    "plt.title(f\"Distribution of {target_col}\")\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Correlation Matrix\n",
                    "plt.figure(figsize=(8, 6))\n",
                    "sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')\n",
                    "plt.title(\"Correlation Heatmap\")\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Run Pipeline"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "metrics_df = run_dataset_pipeline('uci')\n",
                    "metrics_df"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open("notebooks/uci.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2)
    logging.info("Created notebooks/uci.ipynb")

def main():
    logging.basicConfig(level=logging.INFO)
    create_california_notebook()
    create_ames_notebook()
    create_uci_notebook()

if __name__ == '__main__':
    main()
