# utils/data_loader.py
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

def load_data(csv_path):
    """
    Carga el CSV procesado, filtra fechas y devuelve el dataframe escalado y las fechas para secuencias.
    """
    data = pd.read_csv(csv_path, compression='gzip', parse_dates=['Date'], index_col='Date')
    
    # Filtrar rango temporal 2005-2025
    start_date = "2005-01-01"
    end_date = "2025-10-31"
    data = data.loc[start_date:end_date]
    
    # Columnas principales
    pca_cols = [col for col in data.columns if "PCA" in col]
    main_cols = ['BBVA.MC_Close', 'SAN.MC_Close'] + pca_cols
    data_rnn = data[main_cols].ffill().bfill()
    
    return data_rnn, data_rnn.index

def load_model_and_scaler(model_path, scaler_path):
    """
    Carga modelo LSTM y scaler.
    """
    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    return model, scaler
