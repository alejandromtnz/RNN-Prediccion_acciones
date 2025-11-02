# EVENTOS HISTÓRICOS SIGNIFICATIVOS
# src/data_loader_events.py

import pandas as pd
from pathlib import Path

# Carpeta donde guardaremos los eventos
EVENTS_PATH = Path("data/raw/events_data/")
EVENTS_PATH.mkdir(parents=True, exist_ok=True)

# Lista ampliada de eventos históricos (globales, España, BBVA, Santander)
EVENTS = [
    {"name": "Guerra del Golfo", "start": "1990-08-02", "end": "1991-02-28", "impact": 2},
    {"name": "Tratado de Maastricht / Integración UE", "start": "1992-02-07", "end": "1993-11-01", "impact": 2},
    {"name": "Juegos Olímpicos Barcelona 1992 (impacto económico España)", "start": "1992-07-25", "end": "1992-08-09", "impact": 1},
    {"name": "Crisis del peso mexicano", "start": "1994-12-20", "end": "1995-03-31", "impact": 2},
    {"name": "Crisis financiera Asiática", "start": "1997-07-02", "end": "1998-12-31", "impact": 3},
    {"name": "Crisis rusa / default deuda (Rusia)", "start": "1998-08-17", "end": "1998-08-17", "impact": 2},
    {"name": "Fusión BBV + Argentaria -> Creación de BBVA", "start": "1999-10-19", "end": "1999-10-19", "impact": 2},
    {"name": "Introducción del euro (lanzamiento y entrada)", "start": "1999-01-01", "end": "2002-01-01", "impact": 3},
    {"name": "Burbuja punto.com", "start": "2000-03-01", "end": "2002-03-01", "impact": 2},
    {"name": "Expansión internacional BBVA en América Latina", "start": "2000-01-01", "end": "2005-12-31", "impact": 2},
    {"name": "Atentados 11S (Nueva York)", "start": "2001-09-11", "end": "2001-09-30", "impact": 3},
    {"name": "Santander adquiere Abbey National (Reino Unido)", "start": "2004-09-15", "end": "2004-09-15", "impact": 2},
    {"name": "Burbuja inmobiliaria España / boom y corrección", "start": "2006-01-01", "end": "2008-12-31", "impact": 3},
    {"name": "Crisis financiera global", "start": "2007-08-01", "end": "2009-12-31", "impact": 3},
    {"name": "Crisis deuda soberana europea / rescate bancos España", "start": "2010-01-01", "end": "2012-12-31", "impact": 3},
    {"name": "Rescate y reestructuración de Bankia / sector bancario español", "start": "2012-05-01", "end": "2013-12-31", "impact": 3},
    {"name": "Venta y consolidación de activos FROB / Sareb", "start": "2013-01-01", "end": "2014-12-31", "impact": 2},
    {"name": "BBVA compra CatalunyaCaixa", "start": "2014-07-01", "end": "2014-07-31", "impact": 1},
    {"name": "Santander adquiere Banco Popular", "start": "2017-06-07", "end": "2017-06-07", "impact": 3},
    {"name": "Crisis política Cataluña (referéndum 1-O)", "start": "2017-10-01", "end": "2017-10-31", "impact": 1},
    {"name": "COVID-19", "start": "2020-03-01", "end": "2021-06-30", "impact": 3},
    {"name": "Guerra en Ucrania", "start": "2022-02-24", "end": "2025-10-31", "impact": 3},
    {"name": "Guerra de Israel (conflicto Israel-Hamás / Gaza)", "start": "2023-10-07", "end": "2025-10-31", "impact": 2}
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

    # Guardar CSV
    file_path = EVENTS_PATH / "events.csv"
    df_all.to_csv(file_path, index=False)
    print(f"✅ CSV de eventos generado en {file_path}")

if __name__ == "__main__":
    generate_events_csv()
