# utils/preprocessing.py
import numpy as np
import pandas as pd

def create_sequences(data, target_cols=['BBVA.MC_Close','SAN.MC_Close'], lookback=5, horizon=1):
    """
    Crea secuencias para RNN, multi-step compatible.
    """
    X, y = [], []
    for i in range(lookback, len(data)-horizon+1):
        X_seq = data.iloc[i-lookback:i].values
        y_seq = data.iloc[i:i+horizon][target_cols].values
        X.append(X_seq)
        y.append(y_seq)
    X, y = np.array(X), np.array(y)
    return X, y

def inverse_scaled(scaled_values, scaler, total_features):
    """
    Desescalado: añade ceros para no afectar las demás features, devuelve solo target_cols.
    """
    padded = np.hstack([scaled_values, np.zeros((scaled_values.shape[0], total_features - scaled_values.shape[1]))])
    return scaler.inverse_transform(padded)[:, :scaled_values.shape[1]]
