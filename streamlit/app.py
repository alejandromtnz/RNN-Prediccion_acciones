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
    page_icon="",
    layout="wide"
)

# ==============================
# 🎨 Estilo corporativo premium global
# ==============================
st.markdown("""
<style>
/* Mantener títulos Cinzel */
h1, h2, h3 {
    font-family: 'Cinzel', serif !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    color: #111;
}

/* Texto general y elementos de Streamlit: fuente predeterminada */
p, li, label, div.stMarkdown, div.stTextInput, div.stSelectbox, div.stButton {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif !important;
    font-weight: 400;
    color: #111;
    line-height: 1.6;
}

/* Sidebar: estilo predeterminado */
section[data-testid="stSidebar"] {
    background-color: #f0f2f6;
    padding-top: 1rem;
}

/* Botones: estilo predeterminado */
div.stButton > button {
    font-family: inherit !important;
    font-weight: 500;
    color: inherit;
    background-color: #e0e0e0;
    border-radius: 4px;
    border: none;
    transition: 0.3s ease all;
}
div.stButton > button:hover {
    background-color: #d0d0d0;
}

/* ===== Selectbox BBVA/Santander: mantener dorado ===== */
div[data-baseweb="select"] > div {
    border: 2px solid transparent !important;
    border-radius: 8px !important;
    transition: all 0.3s ease;
}
div[data-baseweb="select"]:hover > div {
    border-color: #b8860b !important;
    box-shadow: 0 0 10px rgba(184, 134, 11, 0.6);
}
div[data-baseweb="select"] > div:focus-within {
    border-color: #b8860b !important;
    box-shadow: 0 0 12px rgba(184, 134, 11, 0.8);
    background-color: #fffdf3;
}
div[data-baseweb="select"] span {
    font-weight: 600 !important;
    color: #111 !important;
}

/* ===== Radio buttons (si quieres dorado también) ===== */
div[data-baseweb="radio"] label {
    color: #222;
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
        options=["Inicio", "Modelo predictivo", "Perfil profesional"],
        label_visibility="collapsed"
    )

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
    st.title("Proyección bursátil")
    st.markdown("Explora el rendimiento histórico y las proyecciones a 5 días generadas por nuestro modelo RNN.")

    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    # Cargar datos y modelo
    data_path = "../data/processed/final_data.csv.gz"
    data = pd.read_csv(data_path, compression='gzip', parse_dates=['Date'], index_col='Date')
    data = data.loc["2005-01-01":"2025-10-31"]

    pca_cols = [col for col in data.columns if "PCA" in col]
    main_cols = ['BBVA.MC_Close','SAN.MC_Close'] + pca_cols
    data_rnn = data[main_cols].ffill().bfill()

    scaler = joblib.load("../results/models/scaler_lstm_256_128.pkl")
    data_scaled = pd.DataFrame(scaler.transform(data_rnn), columns=data_rnn.columns, index=data_rnn.index)

    model = load_model("../results/models/lstm_256_128_drop0.3_0.2_bs32_final.keras", compile=False)

    lookback = 5
    X_seq, y_seq = create_sequences(data_scaled, lookback=lookback, horizon=1)
    y_seq_reshaped = y_seq.reshape(y_seq.shape[0], y_seq.shape[2])
    dates_all_1 = data_scaled.index[lookback : lookback + len(X_seq)]

    y_pred_scaled = model.predict(X_seq, verbose=0)
    y_pred_inv = inverse_scaled(y_pred_scaled, scaler, data_scaled.shape[1])
    y_real_inv = inverse_scaled(y_seq_reshaped, scaler, data_scaled.shape[1])

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

    bank = st.selectbox("Selecciona entidad:", ["BBVA", "Santander"])
    bank_idx = 0 if bank=="BBVA" else 1

    df_val = pd.DataFrame({
        "date": dates_all_1,
        "y_real": y_real_inv[:, bank_idx],
        "y_pred": y_pred_inv[:, bank_idx]
    })

    last_date = df_val['date'].max()
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_future)
    df_future = pd.DataFrame({"date": future_dates, "pred": future_preds_inv[:, bank_idx]})

    view_option = st.radio("Rango de visualización:", ["Completa", "Último año", "Último mes"], horizontal=True)
    fig = plot_interactive_series(df_val, df_future, view_option)
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# Página: Perfil profesional
# ==============================
elif page == "Perfil profesional":
    st.title("Perfil profesional")
    st.markdown("""
    **Alejandro Martínez Ronda**, analista especializado en modelado predictivo y análisis de datos financieros.  
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
