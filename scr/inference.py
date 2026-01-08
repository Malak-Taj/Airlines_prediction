import pandas as pd
import joblib
import numpy as np

# Load the model & input columns 
model = joblib.load("Models/XGBoost.pkl")
inputs = joblib.load("Metadata/input_columns.pkl")

def make_prediction(values: dict):
    # Predicted instance 
    predicted_instance = pd.DataFrame([values], columns=inputs)
    
    # Feature engineering
    if "Is_Weekend" in inputs:
        predicted_instance["Is_Weekend"] = predicted_instance["Journey_Day"].isin([5, 6]).astype(int)
    
    # model prediction 
    prediction = model.predict(predicted_instance)[0]
    return prediction
