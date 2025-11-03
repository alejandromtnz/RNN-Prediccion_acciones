import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from utils.data_loader import load_data, load_model_and_scaler
from utils.preprocessing import create_sequences, inverse_scaled
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
# Página: Inicio
# ==============================
if page == "Inicio":
    st.title("📈 Predicción de BBVA y Santander")
    st.markdown("Visualiza la predicción histórica y a 5 días usando nuestra RNN entrenada.")

    # ==============================
    # Cargar datos y modelo
    # ==============================
    data_scaled, dates_all_1 = load_data("../../data/processed/final_data.csv.gz")
    model, scaler = load_model_and_scaler(
        "../../results/models/lstm_256_128_drop0.3_0.2_bs32_final.keras",
        "../../results/models/scaler_lstm_256_128.pkl"
    )

    # ==============================
    # Selección de banco
    # ==============================
    bank = st.selectbox("Selecciona banco:", ["BBVA", "Santander"])

    # ==============================
    # Parámetros de predicción
    # ==============================
    lookback = 5
    horizon = 5  # predicción 5 días
    X_seq, y_seq = create_sequences(data_scaled, lookback=lookback, horizon=1)
    y_seq_reshaped = y_seq.reshape(y_seq.shape[0], y_seq.shape[2])

    # Predicción histórica
    y_pred_scaled = model.predict(X_seq)
    y_pred_inv = inverse_scaled(y_pred_scaled, scaler, data_scaled.shape[1])
    y_real_inv = inverse_scaled(y_seq_reshaped, scaler, data_scaled.shape[1])

    # Predicción futura
    last_X = X_seq[-1:].copy()
    n_future = 5
    future_preds_scaled = []
    for _ in range(n_future):
        pred_scaled = model.predict(last_X, verbose=0)
        future_preds_scaled.append(pred_scaled[0])
        new_step = last_X[:, -1, :].copy()
        new_step[0, 0:2] = pred_scaled
        last_X = np.concatenate([last_X[:, 1:, :], new_step.reshape(1,1,-1)], axis=1)
    future_preds_inv = inverse_scaled(np.array(future_preds_scaled), scaler, data_scaled.shape[1])

    # ==============================
    # DataFrame final según banco
    # ==============================
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
