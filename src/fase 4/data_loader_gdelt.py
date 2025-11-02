# src/fase_4/data_loader_news_gnews_seq.py
import requests
import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import time

# =============================
# CONFIG
# =============================
API_KEY = "892906b647f0927e7f1bda47143b81a5"
BASE_URL = "https://gnews.io/api/v4/search"
KEYWORDS = ["BBVA", "Santander", "IBEX", "Banco Central Europeo"]

# Fechas: primero octubre, luego septiembre, etc.
YEARS_MONTHS = [
    ("2025-10-01", "2025-10-31"),
    ("2025-09-01", "2025-09-30"),
    ("2025-08-01", "2025-08-31"),
    ("2025-07-01", "2025-07-31"),
]

NEWS_PATH = Path("data/raw/news_data/")
NEWS_PATH.mkdir(parents=True, exist_ok=True)

analyzer = SentimentIntensityAnalyzer()

# =============================
# FUNCIONES
# =============================
def daterange(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    for n in range((end - start).days + 1):
        yield start + timedelta(n)

def get_news(keyword, date, max_retries=3):
    """Consulta la API de GNews para un keyword y una fecha"""
    params = {
        "q": keyword,
        "from": date,
        "to": date,
        "lang": "es",
        "max": 100,
        "token": API_KEY
    }
    for attempt in range(max_retries):
        try:
            r = requests.get(BASE_URL, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            return [article["title"] for article in data.get("articles", [])]
        except requests.exceptions.HTTPError as e:
            if r.status_code == 429:  # Límite alcanzado
                wait = 10 * (attempt + 1)
                print(f"⚠️ Límite alcanzado para {keyword} en {date}, esperando {wait}s...")
                time.sleep(wait)
            else:
                print(f"❌ Error HTTP {r.status_code} en {keyword} {date}: {e}")
                return []
        except Exception as e:
            print(f"❌ Error al consultar {keyword} en {date}: {e}")
            time.sleep(5)
    return []

def compute_sentiment(headlines):
    """Calcula sentimiento promedio de una lista de headlines"""
    if not headlines:
        return 0
    scores = []
    for hl in headlines:
        vs = analyzer.polarity_scores(hl)
        if vs["compound"] >= 0.05:
            scores.append(1)
        elif vs["compound"] <= -0.05:
            scores.append(-1)
        else:
            scores.append(0)
    avg = sum(scores)/len(scores)
    if avg > 0:
        return 1
    elif avg < 0:
        return -1
    else:
        return 0

# =============================
# MAIN
# =============================
def main():
    all_data = []

    for start_date, end_date in YEARS_MONTHS:
        print(f"🔹 Procesando del {start_date} al {end_date}...")
        for single_date in daterange(start_date, end_date):
            date_str = single_date.strftime("%Y-%m-%d")
            headlines = []

            # CONSULTA SECUENCIAL: una keyword a la vez
            for kw in KEYWORDS:
                news = get_news(kw, date_str)
                headlines.extend(news)
                time.sleep(2)  # espera 2 segundos entre consultas

            score = compute_sentiment(headlines)
            print(f"{date_str} | Headlines: {len(headlines)} | Sentiment: {score}")

            all_data.append({
                "Date": date_str,
                "Sentiment_Score": score,
                "Num_Headlines": len(headlines)
            })

            # Guardar parcial cada día
            pd.DataFrame(all_data).to_csv(NEWS_PATH / "news_sentiment_partial.csv", index=False)

    # Guardar CSV final
    pd.DataFrame(all_data).to_csv(NEWS_PATH / "news_sentiment_all.csv", index=False)
    print(f"✅ CSV final guardado en {NEWS_PATH / 'news_sentiment_all.csv'}")

if __name__ == "__main__":
    main()






#################################################



# src/fase_4/data_loader_news_gnews_seq.py
import requests
import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import time

# =============================
# CONFIG
# =============================
API_KEY = "892906b647f0927e7f1bda47143b81a5"
BASE_URL = "https://gnews.io/api/v4/search"
KEYWORDS = ["BBVA", "Santander", "IBEX", "Banco Central Europeo"]

# Fechas: primero octubre, luego septiembre, etc.
YEARS_MONTHS = [
    ("2025-10-01", "2025-10-31"),
    ("2025-09-01", "2025-09-30"),
    ("2025-08-01", "2025-08-31"),
    ("2025-07-01", "2025-07-31"),
]

NEWS_PATH = Path("data/raw/news_data/")
NEWS_PATH.mkdir(parents=True, exist_ok=True)

analyzer = SentimentIntensityAnalyzer()

# =============================
# FUNCIONES
# =============================
def daterange(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    for n in range((end - start).days + 1):
        yield start + timedelta(n)

def get_news(keyword, date, max_retries=3):
    """Consulta la API de GNews para un keyword y una fecha"""
    params = {
        "q": keyword,
        "from": date,
        "to": date,
        "lang": "es",
        "max": 100,
        "token": API_KEY
    }
    for attempt in range(max_retries):
        try:
            r = requests.get(BASE_URL, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            titles = [article["title"] for article in data.get("articles", [])]
            # eliminar duplicados
            return list(dict.fromkeys(titles))
        except requests.exceptions.HTTPError as e:
            if r.status_code == 429:  # Límite alcanzado
                wait = 10 * (attempt + 1)
                print(f"⚠️ Límite alcanzado para {keyword} en {date}, esperando {wait}s...")
                time.sleep(wait)
            else:
                print(f"❌ Error HTTP {r.status_code} en {keyword} {date}: {e}")
                return []
        except Exception as e:
            print(f"❌ Error al consultar {keyword} en {date}: {e}")
            time.sleep(5)
    return []

def compute_sentiment(headlines):
    """Calcula sentimiento promedio de una lista de headlines"""
    if not headlines:
        return 0
    scores = []
    for hl in headlines:
        vs = analyzer.polarity_scores(hl)
        if vs["compound"] >= 0.05:
            scores.append(1)
        elif vs["compound"] <= -0.05:
            scores.append(-1)
        else:
            scores.append(0)
    avg = sum(scores)/len(scores)
    if avg > 0:
        return 1
    elif avg < 0:
        return -1
    else:
        return 0

# =============================
# MAIN
# =============================
def main():
    all_data = []

    for start_date, end_date in YEARS_MONTHS:
        print(f"🔹 Proces
