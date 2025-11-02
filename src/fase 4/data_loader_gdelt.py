import requests
import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

API_KEY = "892906b647f0927e7f1bda47143b81a5"
BASE_URL = "https://gnews.io/api/v4/search"
KEYWORDS = ["BBVA", "Santander", "IBEX", "Banco Central Europeo"]

NEWS_PATH = Path("data/raw/news_data/")
NEWS_PATH.mkdir(parents=True, exist_ok=True)

analyzer = SentimentIntensityAnalyzer()

def get_news(keyword):
    """Descarga noticias recientes sobre el keyword (últimos días)."""
    params = {
        "q": keyword,
        "lang": "es",
        "max": 100,
        "token": API_KEY,
        "sortby": "publishedAt"
    }
    try:
        r = requests.get(BASE_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        return [a["title"] for a in data.get("articles", [])]
    except Exception as e:
        print(f"Error con {keyword}: {e}")
        return []

def compute_sentiment(headlines):
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
    avg = sum(scores) / len(scores)
    return 1 if avg > 0 else -1 if avg < 0 else 0

def main():
    all_data = []
    print("🔹 Recopilando noticias recientes...")
    headlines = []
    for kw in KEYWORDS:
        news = get_news(kw)
        headlines.extend(news)
        time.sleep(2)

    score = compute_sentiment(headlines)
    print(f"Headlines totales: {len(headlines)} | Sentiment promedio: {score}")

    all_data.append({
        "Date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "Sentiment_Score": score,
        "Num_Headlines": len(headlines)
    })

    df = pd.DataFrame(all_data)
    df.to_csv(NEWS_PATH / "news_sentiment_recent.csv", index=False)
    print(f"✅ Guardado en {NEWS_PATH / 'news_sentiment_recent.csv'}")

if __name__ == "__main__":
    main()
