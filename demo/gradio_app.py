import gradio as gr
import torch
import numpy as np
import pandas as pd
import joblib
import sys
import os

# Add the root directory to sys.path otherwise import fails
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.network import ReNN

# --- Constants
MODEL_PATH = "model/model.pth"
SCALER_X_PATH = "model/scaler_X.pkl"
SCALER_Y_PATH = "model/scaler_y.pkl"

# --- Load scalers
if os.path.exists(SCALER_X_PATH) and os.path.exists(SCALER_Y_PATH):
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
else:
    raise FileNotFoundError("Scaler files not found. Please train the model first.")

# --- Dummy input to get input size
input_size = scaler_X.mean_.shape[0]

# --- Load model
model = ReNN(input_size)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("Loaded trained model.")
else:
    print("Model not found. Using untrained model.")

# --- Define prediction function
def predict_yield(rainfall, temperature, elevation, fertilizer, shade, soil_type):
    # One-hot encode soil type
    soil_loamy = 1 if soil_type == "Loamy" else 0
    soil_clay = 1 if soil_type == "Clay" else 0
    
    input_data = np.array([[rainfall, temperature, elevation, fertilizer, shade, soil_loamy, soil_clay]])
    
    # Scale input
    X_scaled = scaler_X.transform(input_data)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        y_pred = model(X_tensor)
    
    # Inverse transform prediction
    y_unscaled = scaler_y.inverse_transform(y_pred.numpy())
    
    return round(float(y_unscaled[0][0]), 2)

inputs = [
    gr.Slider(500, 3000, label="Rainfall (mm/year)"),
    gr.Slider(10, 35, label="Avg Temperature (°C)"),
    gr.Slider(500, 2000, label="Elevation (m)"),
    gr.Slider(0, 100, label="Fertilizer (kg/ha)"),
    gr.Slider(0, 100, label="Shade Trees (%)"),
    gr.Dropdown(["Loamy", "Clay"], label="Soil Type"),
]

output = gr.Number(label="Predicted Yield (kg/ha)")

demo = gr.Interface(fn=predict_yield, inputs=inputs, outputs=output, title="YieldBrew: Coffee Yield Estimator")
demo.launch()
