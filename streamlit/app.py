import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.plotting import plot_interactive_series

st.set_page_config(
    page_title="RNN Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# ==============================
# Sidebar: Navegación
# ==============================
st.sidebar.title("Menú")
page = st.sidebar.radio("Ir a:", ["Inicio", "Sobre mi", "Sobre el modelo"])

# ==============================
# Funciones auxiliares
# ==============================
def create_sequences(data, target_cols=['BBVA.MC_Close','SAN.MC_Close'], lookback=5, horizon=1):
    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1):
        X_seq = data.iloc[i - lookback:i].values
        y_seq = data.iloc[i:i + horizon][target_cols].values
        X.append(X_seq)
        y.append(y_seq)
    return np.array(X), np.array(y)

def inverse_scaled(scaled_values, scaler, total_features):
    padded = np.hstack([scaled_values, np.zeros((scaled_values.shape[0], total_features - scaled_values.shape[1]))])
    return scaler.inverse_transform(padded)[:, :scaled_values.shape[1]]

# ==============================
# Página: Inicio
# ==============================
if page == "Inicio":
    st.title("📈 Predicción de BBVA y Santander")
    st.markdown("Visualiza la predicción histórica y a 5 días usando nuestra RNN entrenada.")

    # ==============================
    # Fijar semilla para reproducibilidad
    # ==============================
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    # ==============================
    # Cargar datos y modelo
    # ==============================
    data_path = "../data/processed/final_data.csv.gz"
    data = pd.read_csv(data_path, compression='gzip', parse_dates=['Date'], index_col='Date')

    # Filtrar rango temporal
    data = data.loc["2005-01-01":"2025-10-31"]

    # Columnas principales
    pca_cols = [col for col in data.columns if "PCA" in col]
    main_cols = ['BBVA.MC_Close','SAN.MC_Close'] + pca_cols
    data_rnn = data[main_cols].ffill().bfill()

    # Escalado
    scaler = joblib.load("../results/models/scaler_lstm_256_128.pkl")
    data_scaled = pd.DataFrame(scaler.transform(data_rnn), columns=data_rnn.columns, index=data_rnn.index)

    # Cargar modelo
    model = load_model("../results/models/lstm_256_128_drop0.3_0.2_bs32_final.keras", compile=False)

    # ==============================
    # Crear secuencias
    # ==============================
    lookback = 5
    X_seq, y_seq = create_sequences(data_scaled, lookback=lookback, horizon=1)
    y_seq_reshaped = y_seq.reshape(y_seq.shape[0], y_seq.shape[2])

    # Vector de fechas correspondiente a cada secuencia
    dates_all_1 = data_scaled.index[lookback : lookback + len(X_seq)]

    # ==============================
    # Predicción histórica
    # ==============================
    y_pred_scaled = model.predict(X_seq, verbose=0)
    y_pred_inv = inverse_scaled(y_pred_scaled, scaler, data_scaled.shape[1])
    y_real_inv = inverse_scaled(y_seq_reshaped, scaler, data_scaled.shape[1])

    # ==============================
    # Predicción futura (n_future días)
    # ==============================
    n_future = 5
    last_X = X_seq[-1:].copy()
    future_preds_scaled = []

    for _ in range(n_future):
        pred_scaled = model.predict(last_X, verbose=0)
        future_preds_scaled.append(pred_scaled[0])
        new_step = last_X[:, -1, :].copy()
        new_step[0, 0:2] = pred_scaled
        last_X = np.concatenate([last_X[:, 1:, :], new_step.reshape(1,1,-1)], axis=1)

    future_preds_inv = inverse_scaled(np.array(future_preds_scaled), scaler, data_scaled.shape[1])

    # ==============================
    # DataFrame final por banco
    # ==============================
    bank = st.selectbox("Selecciona banco:", ["BBVA", "Santander"])
    bank_idx = 0 if bank=="BBVA" else 1

    df_val = pd.DataFrame({
        "date": dates_all_1,
        "y_real": y_real_inv[:, bank_idx],
        "y_pred": y_pred_inv[:, bank_idx]
    })

    last_date = df_val['date'].max()
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_future)
    df_future = pd.DataFrame({"date": future_dates, "pred": future_preds_inv[:, bank_idx]})

    # ==============================
    # Opciones de visualización
    # ==============================
    view_option = st.radio("Ver serie:", ["Completa", "Mensual", "Semanal"])
    fig = plot_interactive_series(df_val, df_future, view_option)
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# Página: Sobre mi
# ==============================
elif page == "Sobre mi":
    st.title("Sobre mi")
    st.markdown("""
    Soy Alejandro Martínez Ronda, estudiante de Ingeniería Matemática con experiencia en análisis de datos y modelos predictivos.
    
    - [GitHub](https://github.com/tu_usuario)
    - Contacto: tu.email@example.com
    """)

# ==============================
# Página: Sobre el modelo
# ==============================
elif page == "Sobre el modelo":
    st.title("Sobre el modelo")
    st.markdown("""
    - Modelo: LSTM con 2 capas (256 y 128 unidades) y dropout (0.3, 0.2)
    - Escalado: StandardScaler
    - Datos: Precios de BBVA y Santander + componentes PCA
    - Horizonte de predicción: 5 días
    - Repositorio del modelo: [GitHub](https://github.com/tu_usuario/tu_repositorio)
    """)
