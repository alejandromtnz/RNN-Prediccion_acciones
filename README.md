# Stock Price Prediction with Recurrent Neural Networks (LSTM)

Deep learning project that forecasts the price evolution of banking-sector stocks
(BBVA, Santander, …) using **LSTM** recurrent neural networks. The model is
trained on market data enriched with macroeconomic indicators, foreign exchange,
commodities, market-risk measures, historical events and news sentiment. It ships
with an interactive **Streamlit** app to explore the predictions.

> Built as a final-degree / machine-learning project. Its purpose is educational
> and research-oriented — **it is not investment advice**.

## Demo

The Streamlit app lets you pick an asset, run the trained model and visualise the
historical series together with a multi-day forecast.

```bash
streamlit run streamlit/app.py
```

## Repository structure

```
.
├── data/
│   ├── raw/                 # Original per-category data (unprocessed)
│   │   ├── stock_data/          # Quotes (BBVA, Santander, indices…)
│   │   ├── commodity_data/      # Commodities (Brent, Gold, Copper…)
│   │   ├── fx_data/             # Exchange rates (EURUSD, DXY…)
│   │   ├── macro_data/          # Macro (CPI, GDP, unemployment, M2/M3…)
│   │   ├── market_risk/         # Market risk (VIX, EVZ…)
│   │   ├── events_data/         # Historical high-impact events
│   │   └── news_data/           # News sentiment
│   └── processed/           # Clean, merged data and final feature set
├── src/                     # Data download / ingestion scripts, by phase
│   ├── fase 2/                  # Market, commodities/FX and macro loaders
│   ├── fase 3/                  # Events loader
│   └── fase 4/                  # GDELT (news) loader
├── notebooks/
│   ├── fases-info.ipynb         # Index / description of the pipeline phases
│   ├── data_clean/              # Data cleaning and merging (phase 5)
│   └── RNN/                     # Modelling and prediction
│       ├── 1-modelo.ipynb           # LSTM training
│       ├── 2-predicciones.ipynb     # Prediction generation
│       └── pruebas/                 # Experimental notebooks (drafts)
├── results/
│   └── models/              # Trained LSTM models (.keras) and scalers (.pkl)
├── streamlit/               # Web application (self-contained for deployment)
│   ├── app.py
│   ├── utils/                   # Data loading, preprocessing and plotting
│   ├── data/ · results/         # Copy of the data/models used by the app
│   └── requirements_streamlit.txt
├── requirements.txt
└── .devcontainer/           # GitHub Codespaces configuration
```

## Pipeline phases

1. **Data ingestion** (`src/`): download of quotes (yfinance), commodities, FX,
   macroeconomic indicators, events and news.
2. **Cleaning and merging** (`notebooks/data_clean/`): all sources merged into a
   single date-aligned dataset and feature selection.
3. **Modelling** (`notebooks/RNN/1-modelo.ipynb`): building and training the LSTM
   (256→128 architecture with dropout, configurable lookback window).
4. **Prediction** (`notebooks/RNN/2-predicciones.ipynb`): multi-day forecast on
   recent data.
5. **Visualisation** (`streamlit/app.py`): interactive interface.

## Requirements & installation

Requires **Python 3.11**.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

To run only the Streamlit app, a lighter environment is enough:

```bash
pip install -r streamlit/requirements_streamlit.txt
```

## Usage

**Reproduce training**

```bash
jupyter lab            # open notebooks/RNN/1-modelo.ipynb
```

**Launch the app**

```bash
streamlit run streamlit/app.py
```

It can also be opened directly in **GitHub Codespaces**: the `.devcontainer`
starts the app automatically on port 8501.

## Data & model

- Data compressed as `.csv.gz` under `data/`.
- Main model: `results/models/lstm_256_128_drop0.3_0.2_bs32_final.keras`
  with its scaler `results/models/scaler_lstm_256_128.pkl`.

## Disclaimer

Educational and research content only. The predictions must not be used to make
real investment decisions.

---

> **Nota:** el código, los notebooks y los datos están en español. Este README
> está en inglés para mayor alcance; la estructura de carpetas conserva sus
> nombres originales.
