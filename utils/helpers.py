import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

def load_model(model_path):
    """Load a saved model from disk"""
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {str(e)}")
        return None

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

def prepare_features(input_df, feature_columns):
    """Prepare features for prediction"""
    # Ensure all required columns are present
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Fill missing with zeros
    
    return input_df[feature_columns]

def generate_future_dates(start_date, periods=30, freq='D'):
    """Generate future dates for forecasting"""
    return pd.date_range(
        start=start_date,
        periods=periods,
        freq=freq
    )
