# Predicción de acciones con redes neuronales recurrentes (LSTM)

Proyecto de predicción de la evolución de precios de acciones del sector bancario
(BBVA, Santander, …) mediante redes neuronales recurrentes tipo **LSTM**. El
modelo se entrena a partir de datos de mercado enriquecidos con variables
macroeconómicas, divisas, materias primas, riesgo de mercado, eventos históricos
y sentimiento de noticias. Incluye una aplicación **Streamlit** interactiva para
explorar las predicciones.

> Proyecto desarrollado como Trabajo de Fin de Grado / práctica de aprendizaje
> automático. El objetivo es didáctico y de investigación; **no constituye una
> recomendación de inversión**.

## Demo

La aplicación Streamlit permite seleccionar un activo, lanzar el modelo entrenado
y visualizar la serie histórica junto con la predicción a varios días.

```bash
streamlit run streamlit/app.py
```

## Estructura del repositorio

```
.
├── data/
│   ├── raw/                 # Datos originales por categoría (sin procesar)
│   │   ├── stock_data/          # Cotizaciones (BBVA, Santander, índices…)
│   │   ├── commodity_data/      # Materias primas (Brent, Oro, Cobre…)
│   │   ├── fx_data/             # Tipos de cambio (EURUSD, DXY…)
│   │   ├── macro_data/          # Macro (CPI, GDP, paro, M2/M3…)
│   │   ├── market_risk/         # Riesgo de mercado (VIX, EVZ…)
│   │   ├── events_data/         # Eventos históricos con impacto
│   │   └── news_data/           # Sentimiento de noticias
│   └── processed/           # Datos limpios, unificados y features finales
├── src/                     # Scripts de descarga/ingesta de datos por fases
│   ├── fase 2/                  # Loaders de mercado, commodities/FX y macro
│   ├── fase 3/                  # Loader de eventos
│   └── fase 4/                  # Loader GDELT (noticias)
├── notebooks/
│   ├── fases-info.ipynb         # Índice / descripción de las fases del pipeline
│   ├── data_clean/              # Limpieza y unificación de datos (fase 5)
│   └── RNN/                     # Modelado y predicción
│       ├── 1-modelo.ipynb           # Entrenamiento del LSTM
│       ├── 2-predicciones.ipynb     # Generación de predicciones
│       └── pruebas/                 # Notebooks experimentales (borradores)
├── results/
│   └── models/              # Modelos LSTM entrenados (.keras) y scalers (.pkl)
├── streamlit/               # Aplicación web (autocontenida para despliegue)
│   ├── app.py
│   ├── utils/                   # Carga de datos, preprocesado y gráficas
│   ├── data/ · results/         # Copia de los datos/modelos que usa la app
│   └── requirements_streamlit.txt
├── requirements.txt
└── .devcontainer/           # Configuración de GitHub Codespaces
```

## Fases del pipeline

1. **Ingesta de datos** (`src/`): descarga de cotizaciones (yfinance), materias
   primas, divisas, indicadores macroeconómicos, eventos y noticias.
2. **Limpieza y unificación** (`notebooks/data_clean/`): fusión de todas las
   fuentes en un único dataset alineado por fecha y selección de variables.
3. **Modelado** (`notebooks/RNN/1-modelo.ipynb`): construcción y entrenamiento
   del LSTM (arquitectura 256→128 con dropout, ventana temebral configurable).
4. **Predicción** (`notebooks/RNN/2-predicciones.ipynb`): predicción a varios
   días sobre datos recientes.
5. **Visualización** (`streamlit/app.py`): interfaz interactiva.

## Requisitos e instalación

Requiere **Python 3.11**.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Para ejecutar únicamente la app Streamlit basta con un entorno más ligero:

```bash
pip install -r streamlit/requirements_streamlit.txt
```

## Uso

**Reproducir el entrenamiento**

```bash
jupyter lab            # abrir notebooks/RNN/1-modelo.ipynb
```

**Lanzar la aplicación**

```bash
streamlit run streamlit/app.py
```

También puede abrirse directamente en **GitHub Codespaces**: el `.devcontainer`
arranca la app automáticamente en el puerto 8501.

## Datos y modelo

- Datos comprimidos en `.csv.gz` dentro de `data/`.
- Modelo principal: `results/models/lstm_256_128_drop0.3_0.2_bs32_final.keras`
  con su scaler `results/models/scaler_lstm_256_128.pkl`.

## Aviso

Contenido con fines educativos y de investigación. Las predicciones no deben
utilizarse para tomar decisiones de inversión reales.
