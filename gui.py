"""
Multi-Dataset & Multi-Model Interactive Web GUI
===============================================
Gradio Web Interface allowing live price predictions across all 3 datasets
(California, Ames, UCI Real Estate) and all 6 trained regression models.

Usage:
    python gui.py
"""

import os
import sys
import numpy as np
import pandas as pd
import gradio as gr

sys.path.append(os.path.abspath("src"))

from preprocessing import load_california_data, load_ames_data, load_uci_data
from models import load_model

MODEL_KEY_MAP = {
    "Simple Linear Regression": "simple_linear",
    "Multiple Linear Regression": "multiple_linear",
    "Generalized Linear Model (GLM/GI)": "glm",
    "Decision Tree Regressor": "decision_tree",
    "Random Forest Regressor": "random_forest",
    "Gradient Boosting Regressor": "gradient_boosting"
}

# --- 1. California Prediction Handler ---
def predict_california(model_name, med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude):
    model_key = MODEL_KEY_MAP[model_name]
    preproc_path = "outputs/models/california_preprocessor.joblib"
    model_path = f"outputs/models/california_{model_key}.joblib"
    
    if not os.path.exists(preproc_path) or not os.path.exists(model_path):
        return f"Error: Model file '{model_path}' not found. Please run `python retrain.py --dataset california`."
        
    preprocessor = load_model(preproc_path)
    model = load_model(model_path)
    
    input_df = pd.DataFrame([{
        'MedInc': med_inc,
        'HouseAge': house_age,
        'AveRooms': ave_rooms,
        'AveBedrms': ave_bedrms,
        'Population': population,
        'AveOccup': ave_occup,
        'Latitude': latitude,
        'Longitude': longitude
    }])
    
    proc_input = preprocessor.transform(input_df)
    if model_key == 'simple_linear':
        proc_input = proc_input[:, 0].reshape(-1, 1)
        
    pred = model.predict(proc_input)[0]
    val_dollars = pred * 100000
    return f"${val_dollars:,.2f} USD  (Raw Index: {pred:.4f})"


# --- 2. Ames Prediction Handler ---
def predict_ames(model_name, gr_liv_area, overall_qual, total_bsmt_sf, garage_cars, year_built, year_remod):
    model_key = MODEL_KEY_MAP[model_name]
    preproc_path = "outputs/models/ames_preprocessor.joblib"
    model_path = f"outputs/models/ames_{model_key}.joblib"
    
    if not os.path.exists(preproc_path) or not os.path.exists(model_path):
        return f"Error: Model file '{model_path}' not found. Please run `python retrain.py --dataset ames`."
        
    preprocessor = load_model(preproc_path)
    model = load_model(model_path)
    
    # Load dataset sample to get template row with median defaults for remaining 74 features
    df, target_col = load_ames_data()
    template = df.drop(columns=[target_col]).head(1).copy()
    if 'Id' in template.columns:
        template = template.drop(columns=['Id'])
        
    # Override user-specified key features
    template['GrLivArea'] = gr_liv_area
    template['OverallQual'] = overall_qual
    template['TotalBsmtSF'] = total_bsmt_sf
    template['GarageCars'] = garage_cars
    template['YearBuilt'] = year_built
    template['YearRemodAdd'] = year_remod
    
    proc_input = preprocessor.transform(template)
    if model_key == 'simple_linear':
        proc_input = proc_input[:, 0].reshape(-1, 1)
        
    pred_fit = model.predict(proc_input)[0]
    # Ames target was log-transformed (log1p)
    pred_price = np.expm1(pred_fit)
    return f"${pred_price:,.2f} USD"


# --- 3. UCI Real Estate Prediction Handler ---
def predict_uci(model_name, trans_date, house_age, distance_mrt, conv_stores, latitude, longitude):
    model_key = MODEL_KEY_MAP[model_name]
    preproc_path = "outputs/models/uci_preprocessor.joblib"
    model_path = f"outputs/models/uci_{model_key}.joblib"
    
    if not os.path.exists(preproc_path) or not os.path.exists(model_path):
        return f"Error: Model file '{model_path}' not found. Please run `python retrain.py --dataset uci`."
        
    preprocessor = load_model(preproc_path)
    model = load_model(model_path)
    
    input_df = pd.DataFrame([{
        'X1 transaction date': trans_date,
        'X2 house age': house_age,
        'X3 distance to the nearest MRT station': distance_mrt,
        'X4 number of convenience stores': conv_stores,
        'X5 latitude': latitude,
        'X6 longitude': longitude
    }])
    
    proc_input = preprocessor.transform(input_df)
    if model_key == 'simple_linear':
        proc_input = proc_input[:, 0].reshape(-1, 1)
        
    pred_unit_price = model.predict(proc_input)[0]
    return f"{pred_unit_price:.2f} 10000 NTD/Ping  (Unit Price)"


# --- Gradio Multi-Dataset Dashboard Interface ---
model_choices = list(MODEL_KEY_MAP.keys())

with gr.Blocks(title="🏡 Universal House Price Prediction Dashboard") as demo:
    gr.Markdown("# 🏡 Universal House Price Prediction Dashboard")
    gr.Markdown("Select a dataset tab below, choose any trained regression model, and adjust inputs to compute live predictions.")
    
    # --- Tab 1: California ---
    with gr.Tab("California Housing"):
        gr.Markdown("### California Housing Dataset (20,640 records)")
        cal_model = gr.Dropdown(choices=model_choices, value="Gradient Boosting Regressor", label="Select Regression Algorithm")
        with gr.Row():
            with gr.Column():
                med_inc = gr.Slider(0.5, 15.0, value=3.87, label="Median Income (in $10,000s)")
                house_age = gr.Slider(1.0, 52.0, value=28.0, label="House Age (years)")
                ave_rooms = gr.Slider(1.0, 10.0, value=5.4, label="Average Rooms")
                ave_bedrms = gr.Slider(0.5, 5.0, value=1.0, label="Average Bedrooms")
            with gr.Column():
                population = gr.Slider(100, 5000, value=1425, label="Population")
                ave_occup = gr.Slider(1.0, 6.0, value=3.0, label="Average Occupants")
                latitude = gr.Slider(32.0, 42.0, value=35.6, label="Latitude")
                longitude = gr.Slider(-124.0, -114.0, value=-119.5, label="Longitude")
        cal_btn = gr.Button("Predict Price (California)", variant="primary")
        cal_output = gr.Textbox(label="Predicted Valuation")
        
        cal_btn.click(
            fn=predict_california,
            inputs=[cal_model, med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude],
            outputs=cal_output
        )

    # --- Tab 2: Ames ---
    with gr.Tab("Ames Housing (Iowa)"):
        gr.Markdown("### Ames Housing Dataset (1,460 records, 80 features)")
        ames_model = gr.Dropdown(choices=model_choices, value="Gradient Boosting Regressor", label="Select Regression Algorithm")
        with gr.Row():
            with gr.Column():
                gr_liv_area = gr.Slider(500, 4000, value=1500, label="Above Grade Living Area (sq ft)")
                overall_qual = gr.Slider(1, 10, value=6, step=1, label="Overall Quality (1-10 scale)")
                total_bsmt_sf = gr.Slider(0, 3000, value=1000, label="Total Basement Area (sq ft)")
            with gr.Column():
                garage_cars = gr.Slider(0, 4, value=2, step=1, label="Garage Capacity (Cars)")
                year_built = gr.Slider(1872, 2010, value=1973, step=1, label="Original Construction Year")
                year_remod = gr.Slider(1950, 2010, value=1994, step=1, label="Remodel / Addition Year")
        ames_btn = gr.Button("Predict Price (Ames)", variant="primary")
        ames_output = gr.Textbox(label="Predicted Valuation")
        
        ames_btn.click(
            fn=predict_ames,
            inputs=[ames_model, gr_liv_area, overall_qual, total_bsmt_sf, garage_cars, year_built, year_remod],
            outputs=ames_output
        )

    # --- Tab 3: UCI Real Estate ---
    with gr.Tab("UCI Real Estate (Taiwan)"):
        gr.Markdown("### UCI Real Estate Valuation Dataset (414 records, Taiwan)")
        uci_model = gr.Dropdown(choices=model_choices, value="Random Forest Regressor", label="Select Regression Algorithm")
        with gr.Row():
            with gr.Column():
                trans_date = gr.Slider(2012.667, 2013.583, value=2013.167, label="Transaction Date (Year.Fraction)")
                uci_house_age = gr.Slider(0.0, 43.8, value=17.7, label="House Age (years)")
                distance_mrt = gr.Slider(23.3, 6488.0, value=1083.8, label="Distance to MRT Station (meters)")
            with gr.Column():
                conv_stores = gr.Slider(0, 10, value=4, step=1, label="Number of Convenience Stores")
                uci_lat = gr.Slider(24.93, 25.01, value=24.97, label="Latitude")
                uci_long = gr.Slider(121.47, 121.57, value=121.53, label="Longitude")
        uci_btn = gr.Button("Predict Price (UCI Real Estate)", variant="primary")
        uci_output = gr.Textbox(label="Predicted Valuation")
        
        uci_btn.click(
            fn=predict_uci,
            inputs=[uci_model, trans_date, uci_house_age, distance_mrt, conv_stores, uci_lat, uci_long],
            outputs=uci_output
        )

if __name__ == '__main__':
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
