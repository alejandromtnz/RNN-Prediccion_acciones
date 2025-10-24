# MACROECONOMIC Y COMPLEMENTARY 

# src/data_loader_macro.py
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

# Ejemplo con FRED y Yahoo Finance
MACRO_VARS = {
    "SP10YYield": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRLTLT01ESM156N",
    "ECB_M3": "https://sdw.ecb.europa.eu/quickviewexport.do?SERIES_KEY=ECB.MFI.M.U2.E.M3.Z1.EUR.A.S.0000.Z01.E&exportType=csv",
    "GDP_ES": "URL_del_CSV_GDP_ES",
    "Inflation_ES": "URL_del_CSV_IPC_ES",
    "Unemployment_ES": "URL_del_CSV_PARO_ES",
    "ECB_rate": "URL_del_CSV_TIPO_BCE",
    "GDP_US": "URL_del_CSV_GDP_US",
    "CPI_US": "URL_del_CSV_CPI_US",
    "Unemployment_US": "URL_del_CSV_PARO_US",
    "M2_EUR": "URL_del_CSV_M2_EUR"
}


def download_macro_csv(name, url):
    print(f"Descargando {name}...")
    try:
        r = requests.get(url)
        if r.status_code != 200:
            print(f"❌ Error descargando {name}")
            return
        df = pd.read_csv(io.StringIO(r.text))
        # Limpiar columnas y renombrar
        if df.shape[1] > 1:
            df.columns = ["Date", name]
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.set_index("Date", inplace=True)
        # Interpolación diaria
        df = df.resample("D").interpolate(method="linear")
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
    # CDS España y Volumen préstamos -> necesitarán fuente manual o Excel
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