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
    st.markdown("Explora el rendimiento histórico y las proyecciones de acciones a 5 días.")

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
    st.markdown("""
        <div style="
            font-size: 13px;
            color: #555;
            text-align: right;
            margin-top: -5px;
            margin-bottom: 5px;
            margin-right: 25px;
        ">
            <i>*Precios ajustados por dividendos y “splits”.*</i>
        </div>
        """, unsafe_allow_html=True)

# ==============================
# Página: Perfil profesional
# ==============================
elif page == "Perfil profesional":
    st.title("Perfil profesional")

    st.markdown(
        """
        <style>
        .business-card {
            background-color: #fdfcf9; /* blanco marfil más claro */
            border: 1px solid #dcdcdc;
            border-radius: 14px;
            padding: 2.5rem 3rem;
            box-shadow: 0 3px 14px rgba(0,0,0,0.08);
            max-width: 900px;
            margin: 2.5rem auto;
            font-size: 1.1rem;
            line-height: 1.75;
        }
        .business-card h2 {
            font-family: 'Cinzel', serif;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #111;
            margin-bottom: 1rem;
        }
        .business-card h3 {
            margin-top: 1.2rem;  /* antes 2rem */
            margin-bottom: 0.2rem;  /* reduce espacio antes de los enlaces */
            font-family: 'Cinzel', serif;
            font-size: 1.05rem;
            color: #222;
            letter-spacing: 0.5px;
        }
        .business-card a {
            color: #c7a008; /* amarillo más suave */
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s ease;
        }
        .business-card a:hover {
            color: #b8860b;
            text-decoration: underline;
        }
        .business-card ul {
            list-style-type: none;
            padding-left: 0;
            margin-top: 0.2rem; /* antes 0.8rem */
            display: flex;
            gap: 1.3rem;
            justify-content: flex-start;
        }
        .business-card li {
            margin-bottom: 0;
        }
        </style>

        <div class="business-card">
            <h2>Alejandro Martínez Ronda</h2>
            <p>
            Analista versátil especializado en <b>modelado predictivo</b>, <b>análisis de datos</b> y <b>aprendizaje automático</b>, en este caso, para el desarrollo de <b>modelos económicos con series temporales</b>.<br><br>
            Combinando una sólida formación en <b>Ingeniería Matemática e Informática</b>, soluciones que integran rigurosidad técnica con visión económica.
            <h3>Contacto:</h3>
            <ul>
                <li><a href="https://github.com/alejandromtnz" target="_blank">GitHub</a></li>
                <li><a href="https://www.linkedin.com/in/alejandromartinezronda/" target="_blank">LinkedIn</a></li>
                <li><a href="mailto:amartron@myuax.com">Correo</a></li>
            </ul>
            </p>
        </div>

        """,
        unsafe_allow_html=True
    )


# ==============================
# Página: Modelo predictivo
# ==============================
elif page == "Modelo predictivo":
    st.title("Modelo predictivo")

    st.markdown("""
    ### Recopilación y estructuración de datos
    El modelo parte de una infraestructura de descarga y almacenamiento automatizada que consolida información económica y financiera desde **1990** hasta la actualidad.  
    Se recopilan más de **300 series históricas** de diversas fuentes:

    - **Acciones e índices bursátiles** (Yahoo Finance): BBVA, Santander, IBEX 35, S&P 500, Euro Stoxx 50, DAX, Nasdaq, MSCI World.  
    - **Commodities y divisas:** Brent, Oro, Gas Natural, Cobre, EUR/USD, DXY.  
    - **Indicadores macroeconómicos:** PIB, inflación, desempleo, tipos de interés (FRED, BCE, Eurostat).  
    - **Riesgos de mercado:** volatilidad implícita (VIX, EVZ) y sectores globales (MSCI Financials).  
    - **Eventos históricos:** crisis, conflictos y shocks financieros desde 1990 clasificados por impacto (1 a 3).

    Cada fuente se almacena con trazabilidad completa bajo una jerarquía versionada (`data/raw/`), lista para actualizaciones y análisis reproducibles.
    """)

    st.markdown("""
    <div style="
        background-color: #f9f9f9;
        border-left: 4px solid #b8860b;
        padding: 10px 15px;
        margin-top: 10px;
        margin-bottom: 10px;
        font-size: 15px;
        line-height: 1.4;
        text-align: justify;
    ">
    <b>NOTA:</b> Los precios de las acciones se emplean en su versión <i>ajustada por dividendos</i> y <i>“splits”</i>, lo que garantiza la coherencia temporal del valor total para el accionista. Por ello, los precios históricos pueden parecer menores (por ejemplo, BBVA en 2005 ≈ 4 € frente a los 13 € nominales originales), ya que se corrigen los efectos de dividendos —pagos que reducen el precio sin reflejar una pérdida real— y divisiones de acciones. 
    Este ajuste permite entrenar el modelo sobre una serie continua y económicamente comparable en el tiempo, eliminando distorsiones provocadas por eventos corporativos que no representan cambios reales en el valor de mercado. De este modo, las variaciones de precio reflejan únicamente la evolución del valor económico de la compañía y no saltos artificiales por ajustes contables, lo que mejora la estabilidad de la serie, la capacidad predictiva de la RNN y la interpretación financiera de los resultados.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Limpieza, integración y reducción dimensional
    Los más de **44 000 registros y 306 variables** pasan por un proceso exhaustivo de depuración:

    - Homogeneización temporal a frecuencia **diaria**.  
    - Normalización y unificación de índices.  
    - Integración de eventos con puntuaciones de impacto económico y geopolítico.

    Posteriormente se calcula la correlación con los precios de **BBVA** y **Santander**, conservando las series más informativas.  
    Tras aplicar **PCA (componentes principales)**, se reducen a unas **90 variables finales** que resumen la dinámica conjunta del mercado.
    """)

    st.markdown("""
    ### Estructura de entrada para la red neuronal
    El modelo utiliza una **ventana temporal de 5 días (lookback)** que captura dependencias a corto plazo.  
    Se aplican dos configuraciones paralelas:

    - **Single-step:** predice el valor del siguiente día.  
    - **Multi-step:** proyecta los próximos 5 días.

    Estructura de entrada:
    ```
    X.shape = (n_muestras, 5, 24)
    y.shape = (n_muestras, horizonte, 2)
    ```
    donde las 24 variables incluyen precios normalizados, componentes PCA y factores macroeconómicos.  
    Todo el conjunto se escala con `StandardScaler` para garantizar estabilidad en el entrenamiento.
    """)

    st.markdown("""
    ### Arquitectura del modelo LSTM
    El núcleo del sistema es una **red neuronal recurrente (RNN)** con celdas **LSTM (Long Short-Term Memory)**, diseñadas para reconocer patrones temporales en datos financieros.

    **Estructura:**
    - LSTM (256 neuronas) — captura tendencias de largo plazo.  
    - Dropout (30%) — reduce sobreajuste.  
    - LSTM (128 neuronas) — sintetiza relaciones de segundo orden.  
    - Dropout (20%) — refuerza generalización.  
    - Capa densa (2 neuronas) — genera las predicciones de **BBVA** y **Santander**.

    Entrenamiento con **Adam (lr = 0.001)** y pérdida **MSE**, monitorizando **MAE** sobre validación.  
    Early Stopping para evita sobreentrenamiento.

    ```
    loss ≈ 0.0109   |   val_loss ≈ 0.0066
    mae  ≈ 0.0648   |   val_mae  ≈ 0.0415
    ```
    """)

    st.markdown("""
    ### Predicción y proyección futura
    El modelo genera dos salidas principales:

    - **Predicción validada:** comparación entre valores reales y estimados en el histórico reciente.  
    - **Proyección futura:** cálculo autoregresivo de los próximos **5 días**, realimentando cada predicción como entrada siguiente.

    Los resultados se reescalan mediante la inversa del `StandardScaler`, obteniendo precios en su escala original.
    """)

    st.markdown("""
    ### Visualización interactiva
    El panel permite explorar:
    - La evolución histórica de precios y predicciones.  
    - Rangos de visualización: *Completa*, *Último año* o *Último mes*.  
    - Comparativas entre los valores reales y los proyectados.

    """)

    st.markdown("""
    ### Resumen técnico
    | Fase | Descripción | Resultado clave |
    |------|--------------|----------------|
    | **Ingesta** | Descarga automática desde Yahoo Finance, FRED y BCE | 306 series originales |
    | **Limpieza y fusión** | Homogeneización diaria, integración de eventos | 44 630 registros |
    | **Selección de variables** | Filtrado por correlación y PCA | 90 variables finales |
    | **Ventana temporal** | Lookback = 5, Horizonte = 1 y 5 | Captura dinámicas de corto plazo |
    | **Modelo LSTM** | 256 → 128 neuronas, Dropout 0.3/0.2 | Val MAE ≈ 0.0415 |
    | **Predicción futura** | Proyección autoregresiva a 5 días | BBVA & Santander forecast |

    **Repositorio del modelo:** [GitHub – RNN-Prediccion_acciones](https://github.com/alejandromtnz/RNN-Prediccion_acciones)
    """)
