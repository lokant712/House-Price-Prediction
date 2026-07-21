"""
Interactive Web GUI - House Price Prediction
================────────────────────────────
Simple Gradio interface allowing interactive browser-based price predictions.

Usage:
    python gui.py
"""

import os
import sys
import numpy as np
import pandas as pd
import gradio as gr

sys.path.append(os.path.abspath("src"))

from preprocessing import load_california_data
from models import load_model

def predict_california_price(med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude):
    preproc_path = "outputs/models/california_preprocessor.joblib"
    model_path = "outputs/models/california_gradient_boosting.joblib"
    
    if not os.path.exists(preproc_path) or not os.path.exists(model_path):
        return "Model files not found. Please run `python retrain.py --dataset california` first."
        
    preprocessor = load_model(preproc_path)
    model = load_model(model_path)
    
    input_data = pd.DataFrame([{
        'MedInc': med_inc,
        'HouseAge': house_age,
        'AveRooms': ave_rooms,
        'AveBedrms': ave_bedrms,
        'Population': population,
        'AveOccup': ave_occup,
        'Latitude': latitude,
        'Longitude': longitude
    }])
    
    proc_input = preprocessor.transform(input_data)
    pred = model.predict(proc_input)[0]
    
    # California target is in $100k units
    predicted_dollars = pred * 100000
    return f"${predicted_dollars:,.2f} USD (Index Value: {pred:.4f})"

# 2-Line Gradio Web GUI Interface Creation & Launch
demo = gr.Interface(
    fn=predict_california_price,
    inputs=[
        gr.Slider(0.5, 15.0, value=3.87, label="Median Income (in $10,000s)"),
        gr.Slider(1.0, 52.0, value=28.0, label="House Age (years)"),
        gr.Slider(1.0, 10.0, value=5.4, label="Average Rooms"),
        gr.Slider(0.5, 5.0, value=1.0, label="Average Bedrooms"),
        gr.Slider(100, 5000, value=1425, label="Population"),
        gr.Slider(1.0, 6.0, value=3.0, label="Average Occupants"),
        gr.Slider(32.0, 42.0, value=35.6, label="Latitude"),
        gr.Slider(-124.0, -114.0, value=-119.5, label="Longitude"),
    ],
    outputs=gr.Textbox(label="Predicted House Valuation"),
    title="🏡 House Price Prediction GUI",
    description="Interactive Predictive Analytics Model Web Dashboard (California Dataset)"
)

if __name__ == '__main__':
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
