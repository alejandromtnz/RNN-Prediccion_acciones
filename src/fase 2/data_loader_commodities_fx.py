# COMMODITIES Y FX (Foreign Exchange)

# src/data_loader_commodities_fx.py
import yfinance as yf
import pandas as pd
from pathlib import Path

# Carpeta donde guardaremos los CSV
COMMODITY_PATH = Path("data/raw/commodity_data/")
COMMODITY_PATH.mkdir(parents=True, exist_ok=True)

FX_PATH = Path("data/raw/fx_data/")
FX_PATH.mkdir(parents=True, exist_ok=True)

# Commodities
COMMODITIES = {
    "Brent": "BZ=F",
    "WTI": "CL=F",
    "Oro": "GC=F",
    "Plata": "SI=F",
    "Cobre": "HG=F",
    "GasNatural": "NG=F",
    "Trigo": "ZW=F",
    "Aluminio": "ALI=F",
    "Hierro": "TIOc1",
    "Uranio": "UX=F"
}

# FX
FX_PAIRS = {
    "EURUSD": "EURUSD=X",
    "EURGBP": "EURGBP=X",
    "USDJPY": "JPY=X",
    "EURCHF": "EURCHF=X",
    "EURJPY": "EURJPY=X",
    "EURBRL": "EURBRL=X",
    "EURMXN": "EURMXN=X",
    "DXY": "DX-Y.NYB"
}

# Fechas de descarga
START_DATE = "1990-01-01"
END_DATE = "2025-10-31"

def download_and_save(ticker_name, ticker_symbol, folder_path):
    """
    Descarga los datos históricos de Yahoo Finance y los guarda en CSV comprimido.
    """
    print(f"Descargando {ticker_name} ({ticker_symbol})...")
    try:
        df = yf.download(ticker_symbol, start=START_DATE, end=END_DATE, progress=False)
        if df.empty:
            print(f"⚠️  No se encontraron datos para {ticker_name}")
            return
        # Guardar CSV comprimido
        file_path = folder_path / f"{ticker_name}.csv.gz"
        df.to_csv(file_path, index=True, compression="gzip")
        print(f"✅ Guardado: {file_path}")
    except Exception as e:
        print(f"❌ Error descargando {ticker_name}: {e}")

def main():
    print("🔹 Descargando Commodities...")
    for name, symbol in COMMODITIES.items():
        download_and_save(name, symbol, COMMODITY_PATH)

    print("🔹 Descargando FX...")
    for name, symbol in FX_PAIRS.items():
        download_and_save(name, symbol, FX_PATH)

    print("🎯 Descarga completada de Commodities y FX.")

if __name__ == "__main__":
    main()
