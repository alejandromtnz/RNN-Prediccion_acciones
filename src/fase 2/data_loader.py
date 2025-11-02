# ACCIONES PRINCIPALES E ÍNDICES BURSÁTILES

# src/data_loader.py
import yfinance as yf
import pandas as pd
from pathlib import Path

# Carpeta donde guardaremos los CSV
RAW_DATA_PATH = Path("data/raw/stock_data/")
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

# Lista de tickers y nombres amigables para guardar
TICKERS = {
    "BBVA": "BBVA.MC",
    "Santander": "SAN.MC",
    "CaixaBank": "CABK.MC",
    "Bankinter": "BKT.MC",
    "Unicaja": "UNI.MC",
    "Sabadell": "SAB.MC",
    "HSBC": "HSBC",
    "DeutscheBank": "DB",
    "JPMorgan": "JPM",
    "Citigroup": "C",
    "IBEX35": "^IBEX",
    "SP500": "^GSPC",
    "EuroStoxx50": "^STOXX50E",
    "NASDAQ": "^IXIC",
    "MSCI_World": "URTH",
    "FTSE100": "^FTSE",
    "DAX": "^GDAXI",
    "CAC40": "^FCHI",
    "DowJones": "^DJI",
    "Nikkei225": "^N225"
}

# Fechas de descarga
START_DATE = "1990-01-01"
END_DATE = "2025-10-31"

def download_and_save(ticker_name, ticker_symbol):
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
        file_path = RAW_DATA_PATH / f"{ticker_name}.csv.gz"
        df.to_csv(file_path, index=True, compression="gzip")
        print(f"✅ Guardado: {file_path}")
    except Exception as e:
        print(f"❌ Error descargando {ticker_name}: {e}")

def main():
    for name, symbol in TICKERS.items():
        download_and_save(name, symbol)
    print("🎯 Descarga completada de acciones e índices.")

if __name__ == "__main__":
    main()
