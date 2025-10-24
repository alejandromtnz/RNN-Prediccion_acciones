# src/fase 2/data_loader_macro.py
import pandas as pd
from pathlib import Path
import yfinance as yf
import requests
import io

# Carpeta donde guardaremos los CSV
MACRO_PATH = Path("data/raw/macro_data/")
MACRO_PATH.mkdir(parents=True, exist_ok=True)

RISK_PATH = Path("data/raw/market_risk/")
RISK_PATH.mkdir(parents=True, exist_ok=True)

# Fechas globales
START_DATE = "1990-01-01"
END_DATE = "2025-10-24"

# =========================
# 1️⃣ Variables macro principales
# =========================

MACRO_VARS = {
    # España y Europa
    "SP10YYield": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRLTLT01ESM156N",
    "ECB_M3": "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.csv",
    # Estados Unidos
    "GDP_US": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDP",
    "CPI_US": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS",
    "Unemployment_US": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE",
    "M2_EUR": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=M2SL"
}

def download_macro_csv(name, url):
    print(f"Descargando {name}...")
    try:
        if "eurostat" in url:
            # Eurostat TSV comprimido
            df = pd.read_csv(url, sep='\t', compression='gzip', encoding='utf-8', skip_blank_lines=True)
            df = df.melt(id_vars=[df.columns[0]], var_name="Date", value_name=name)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.set_index('Date', inplace=True)
        else:
            # FRED y ECB CSV estándar
            r = requests.get(url)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            if "Date" not in df.columns:
                df.rename(columns={df.columns[0]: "Date", df.columns[1]: name}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.set_index('Date', inplace=True)
        
        # Interpolación diaria
        df = df.resample("D").interpolate(method="linear")
        # Guardado
        file_path = MACRO_PATH / f"{name}.csv.gz"
        df.to_csv(file_path, compression="gzip")
        print(f"✅ Guardado {file_path}")
    except Exception as e:
        print(f"❌ Error en {name}: {e}")

# =========================
# 2️⃣ Complementarios / riesgo
# =========================

RISK_TICKERS = {
    "VIX": "^VIX",
    "EVZ": "^EVZ",
    "MSCI_Financials": "IXG"
}

def download_risk_yf(name, ticker):
    print(f"Descargando {name} ({ticker})...")
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if df.empty:
            print(f"⚠️ No se encontraron datos para {name}")
            return
        file_path = RISK_PATH / f"{name}.csv.gz"
        df.to_csv(file_path, compression="gzip")
        print(f"✅ Guardado {file_path}")
    except Exception as e:
        print(f"❌ Error descargando {name}: {e}")

# =========================
# Main
# =========================

def main():
    # Macro variables
    for name, url in MACRO_VARS.items():
        download_macro_csv(name, url)
    # Risk variables
    for name, ticker in RISK_TICKERS.items():
        download_risk_yf(name, ticker)
    print("🎯 Descarga completada de macro y complementarios.")

if __name__ == "__main__":
    main()
