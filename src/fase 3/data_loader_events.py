# EVENTOS HISTÓRICOS SIGNIFICATIVOS

# src/data_loader_events.py
import pandas as pd
from pathlib import Path

# Carpeta donde guardaremos los eventos
EVENTS_PATH = Path("data/raw/events_data/")
EVENTS_PATH.mkdir(parents=True, exist_ok=True)

# Lista de eventos históricos
EVENTS = [
    {"name": "Guerra del Golfo", "start": "1990-08-02", "end": "1991-02-28", "impact": 2},
    {"name": "Burbuja punto.com", "start": "2000-03-01", "end": "2002-03-01", "impact": 2},
    {"name": "11S Atentados NY", "start": "2001-09-11", "end": "2001-09-30", "impact": 3},
    {"name": "Crisis financiera global", "start": "2007-08-01", "end": "2009-12-31", "impact": 3},
    {"name": "Crisis deuda europea", "start": "2010-01-01", "end": "2012-01-01", "impact": 2},
    {"name": "COVID-19", "start": "2020-03-01", "end": "2021-06-30", "impact": 3},
    {"name": "Guerra Ucrania", "start": "2022-02-24", "end": "2024-12-31", "impact": 3}
]

def generate_events_csv():
    all_dates = []

    for event in EVENTS:
        # Generar rango de fechas diario
        dates = pd.date_range(start=event["start"], end=event["end"], freq="D")
        df_event = pd.DataFrame({
            "Date": dates,
            "Event_Name": event["name"],
            "Impact_Score": event["impact"]
        })
        all_dates.append(df_event)

    # Concatenar todos los eventos
    df_all = pd.concat(all_dates, ignore_index=True)
    df_all.sort_values("Date", inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    # Guardar CSV comprimido
    file_path = EVENTS_PATH / "events.csv"
    df_all.to_csv(file_path, index=False)
    print(f"✅ CSV de eventos generado en {file_path}")

if __name__ == "__main__":
    generate_events_csv()
