from xgboost import XGBRegressor
from sklearn.model_selection import cross_validate
import pandas as pd 
import numpy as np
import joblib

# Load the model and data
model = joblib.load("Models/XGBoost.pkl")
 # Load data
df = pd.read_parquet('Data/clean/Cleaned_data.parquet')

# Split features and target
x = df.drop('Price', axis=1)
y_log = np.log1p(df['Price'])



scores = cross_validate(
    model, x, y_log, cv=5,
    scoring={
        'R2': 'r2',
        'MAE': 'neg_mean_absolute_error',
        'RMSE': 'neg_root_mean_squared_error'
    }
)
print("R2:", scores['test_R2'].mean())
print("MAE:", abs(scores['test_MAE']).mean())
print("RMSE:", abs(scores['test_RMSE']).mean())