import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.plotting import plot_interactive_series

# ==============================
# 🏛️ Configuración general
# ==============================
st.set_page_config(
    page_title="RNN Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# ==============================
# 🎨 Estilo corporativo premium global
# ==============================
st.markdown("""
    <style>
    /* =======================
       🎨 Selectbox y Radio refinados
       ======================= */

    /* Selectbox */
    div[data-baseweb="select"] > div {
        border: 1px solid #ccc !important;
        border-radius: 6px !important;
        background-color: #ffffff !important;
        transition: 0.2s ease all;
        padding: 2px 6px !important;
    }
    div[data-baseweb="select"]:hover > div {
        border-color: #b8860b !important;
        box-shadow: 0 0 5px rgba(184,134,11,0.25);
    }

    label[data-baseweb="select"] {
        font-family: 'Cinzel', serif !important;
        color: #111 !important;
        font-weight: 600;
        font-size: 15px !important;
        margin-bottom: 0.1rem !important;
    }

    /* Radios horizontales sin círculos rojos */
    div[data-baseweb="radio"] {
        display: flex !important;
        gap: 1.2rem !important;
        justify-content: flex-start !important;
        align-items: center !important;
        margin-top: 0.3rem !important;
        margin-bottom: 0.8rem !important;
    }

    div[data-baseweb="radio"] label {
        position: relative;
        padding-left: 20px;
        cursor: pointer;
        font-family: 'Lato', sans-serif;
        font-size: 14px;
        color: #222;
        transition: color 0.2s ease;
    }

    div[data-baseweb="radio"] input[type="radio"] {
        appearance: none;
        position: absolute;
        left: 0;
        top: 2px;
        width: 11px;
        height: 11px;
        border: 2px solid #002b5c;
        border-radius: 50%;
        background-color: #fff;
        transition: all 0.2s ease;
    }

    div[data-baseweb="radio"] input[type="radio"]:checked {
        border-color: #b8860b;
        background: radial-gradient(circle, #b8860b 45%, transparent 46%);
    }

    div[data-baseweb="radio"] label:hover input[type="radio"] {
        border-color: #b8860b;
    }

    .stSelectbox, .stRadio {
        margin-top: 0.3rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Ajuste general de espaciado */
    .block-container {
        padding-top: 1.2rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# Sidebar: Navegación
# ==============================
with st.sidebar:
    st.markdown("<h2 style='margin-bottom: 0.5rem;'>Menú</h2>", unsafe_allow_html=True)
    page = st.radio(
        label="",
        options=["Inicio", "Perfil profesional", "Modelo predictivo"],
        label_visibility="collapsed"
    )

# ==============================
# Funciones auxiliares
# ==============================
def create_sequences(data, target_cols=['BBVA.MC_Close', 'SAN.MC_Close'], lookback=5, horizon=1):
    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1):
        X_seq = data.iloc[i - lookback:i].values
        y_seq = data.iloc[i:i + horizon][target_cols].values
        X.append(X_seq)
        y.append(y_seq)
    return np.array(X), np.array(y)

def inverse_scaled(scaled_values, scaler, total_features):
    padded = np.hstack([
        scaled_values,
        np.zeros((scaled_values.shape[0], total_features - scaled_values.shape[1]))
    ])
    return scaler.inverse_transform(padded)[:, :scaled_values.shape[1]]

# ==============================
# Página: Inicio
# ==============================
if page == "Inicio":
    st.title("Proyección bursátil")
    st.markdown("Visualiza el rendimiento histórico y las proyecciones a 5 días generadas por el modelo RNN.")

    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    # Cargar datos y modelo
    data_path = "../data/processed/final_data.csv.gz"
    data = pd.read_csv(data_path, compression='gzip', parse_dates=['Date'], index_col='Date')
    data = data.loc["2005-01-01":"2025-10-31"]

    pca_cols = [col for col in data.columns if "PCA" in col]
    main_cols = ['BBVA.MC_Close', 'SAN.MC_Close'] + pca_cols
    data_rnn = data[main_cols].ffill().bfill()

    scaler = joblib.load("../results/models/scaler_lstm_256_128.pkl")
    data_scaled = pd.DataFrame(
        scaler.transform(data_rnn),
        columns=data_rnn.columns,
        index=data_rnn.index
    )

    model = load_model("../results/models/lstm_256_128_drop0.3_0.2_bs32_final.keras", compile=False)

    lookback = 5
    X_seq, y_seq = create_sequences(data_scaled, lookback=lookback, horizon=1)
    y_seq_reshaped = y_seq.reshape(y_seq.shape[0], y_seq.shape[2])
    dates_all_1 = data_scaled.index[lookback : lookback + len(X_seq)]

    y_pred_scaled = model.predict(X_seq, verbose=0)
    y_pred_inv = inverse_scaled(y_pred_scaled, scaler, data_scaled.shape[1])
    y_real_inv = inverse_scaled(y_seq_reshaped, scaler, data_scaled.shape[1])

    # Predicción futura
    n_future = 3
    last_X = X_seq[-1:].copy()
    future_preds_scaled = []
    for _ in range(n_future):
        pred_scaled = model.predict(last_X, verbose=0)
        future_preds_scaled.append(pred_scaled[0])
        new_step = last_X[:, -1, :].copy()
        new_step[0, 0:2] = pred_scaled
        last_X = np.concatenate([last_X[:, 1:, :], new_step.reshape(1, 1, -1)], axis=1)
    future_preds_inv = inverse_scaled(np.array(future_preds_scaled), scaler, data_scaled.shape[1])

    # Selector de entidad
    bank = st.selectbox("Selecciona entidad:", ["BBVA", "Santander"])
    bank_idx = 0 if bank == "BBVA" else 1

    df_val = pd.DataFrame({
        "date": dates_all_1,
        "y_real": y_real_inv[:, bank_idx],
        "y_pred": y_pred_inv[:, bank_idx]
    })

    last_date = df_val['date'].max()
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_future)
    df_future = pd.DataFrame({"date": future_dates, "pred": future_preds_inv[:, bank_idx]})

    # Selector de rango
    view_option = st.radio(
        "Rango de visualización:",
        ["Completa", "Último año", "Último mes"],
        horizontal=True
    )

    # Gráfico
    fig = plot_interactive_series(df_val, df_future, view_option)
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# Página: Perfil profesional
# ==============================
elif page == "Perfil profesional":
    st.title("Perfil profesional")
    st.markdown("""
    **Alejandro Martínez Ronda**  
    Analista especializado en modelado predictivo y análisis de datos financieros.  
    Experiencia en aprendizaje automático, visualización avanzada y desarrollo de modelos económicos.
    
    **Contacto:**  
    - [GitHub](https://github.com/alejandromtnz)  
    - amartron@myuax.com
    """)

# ==============================
# Página: Modelo predictivo
# ==============================
elif page == "Modelo predictivo":
    st.title("Modelo predictivo")
    st.markdown("""
    **Arquitectura del modelo**  
    - Red neuronal recurrente tipo **LSTM** con 2 capas (256 y 128 unidades).  
    - Regularización mediante *dropout* (0.3, 0.2).  
    - Escalado: *StandardScaler*.  
    - Variables: precios de cierre de **BBVA** y **Santander**, más componentes **PCA**.  
    - Horizonte de predicción: **5 días**.  

    [Repositorio del modelo en GitHub](https://github.com/alejandromtnz/RNN-Prediccion_acciones)
    """)
