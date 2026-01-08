from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from preprocessing import preprocessing_pipeline
import pandas as pd
import numpy as np
import joblib
import os

def train_model():

    os.makedirs("Models", exist_ok=True)
    os.makedirs("Metadata", exist_ok=True)

    # Load data
    df = pd.read_parquet('Data/clean/Cleaned_data.parquet')

    # Split features and target
    x = df.drop('Price', axis=1)
    y_log = np.log1p(df['Price'])

    # Build pipeline
    model_pipeline = Pipeline([
        ('Preprocessing', preprocessing_pipeline()),
        ('Model', XGBRegressor(
            objective='reg:squarederror',
            subsample=0.8,
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            gamma=0,
            colsample_bytree=0.6,
            random_state=42
        ))
    ])

    # Train
    model_pipeline.fit(x, y_log)
    print('Data trained successfully!')

    # Save model
    joblib.dump(model_pipeline, "Models/XGBoost.pkl")

    # Save metadata
    joblib.dump(x.columns.to_list(), 'Metadata/input_columns.pkl')

    unique_values_dict = {col: x[col].unique() for col in x.columns}
    joblib.dump(unique_values_dict, 'Metadata/unique_values.pkl')

if __name__ == '__main__':
    train_model()
