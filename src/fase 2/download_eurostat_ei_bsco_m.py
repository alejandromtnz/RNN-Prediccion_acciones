# MACROECONOMIC DATA FROM EUROSTAT EI_BSCO_M

import requests
import pandas as pd
import zipfile
import io
from pathlib import Path

# Directorio donde se guardarán los datos
DATA_DIR = Path("data/raw/macro_data/")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# URL de la API SDMX 3.0 de Eurostat para el conjunto de datos ei_bsco_m
API_URL = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/ei_bsco_m/all?format=SDMX-CSV"

# Función para descargar y descomprimir el archivo
def download_and_extract(url, dest_folder):
    print(f"Descargando datos desde {url}...")
    response = requests.get(url)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            zf.extractall(dest_folder)
        print(f"Datos extraídos en {dest_folder}")
    else:
        print(f"Error al descargar los datos: {response.status_code}")

# Función para leer y procesar el archivo CSV
def process_csv(file_path):
    print(f"Procesando el archivo {file_path}...")
    df = pd.read_csv(file_path, sep=";", encoding="latin1")
    # Aquí puedes agregar cualquier procesamiento adicional que necesites
    return df

# Función principal
def main():
    download_and_extract(API_URL, DATA_DIR)
    # Asumiendo que el archivo descargado se llama 'ei_bsco_m.csv'
    df = process_csv(DATA_DIR / "ei_bsco_m.csv")
    # Guardar el DataFrame como CSV comprimido
    df.to_csv(DATA_DIR / "ei_bsco_m.csv.gz", index=False, compression="gzip")
    print("Datos procesados y guardados como 'ei_bsco_m.csv.gz'")

if __name__ == "__main__":
    main()
