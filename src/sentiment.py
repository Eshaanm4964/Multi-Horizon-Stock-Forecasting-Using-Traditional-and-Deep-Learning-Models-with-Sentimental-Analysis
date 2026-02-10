import os
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))
analyzer = SentimentIntensityAnalyzer()


def fetch_stock_news(ticker, page_size=8):
    """
    Fetch recent news related to the stock ticker.
    Falls back to general market news if ticker news is unavailable.
    """
    try:
        response = newsapi.get_everything(
            q=ticker,
            language="en",
            sort_by="publishedAt",
            page_size=page_size
        )
        articles = response.get("articles", [])

        # Fallback to general market news if empty
        if not articles:
            response = newsapi.get_everything(
                q="stock market OR finance OR NASDAQ OR S&P 500",
                language="en",
                sort_by="publishedAt",
                page_size=page_size
            )
            articles = response.get("articles", [])

        return articles
    except Exception as e:
        print("NewsAPI error:", e)
        return []


def compute_sentiment_score(articles):
    """
    Compute average VADER compound sentiment score using title + description.
    Returns 0.0 if no valid articles.
    """
    if not articles:
        return 0.0

    scores = []
    for article in articles:
        text = (article.get("title", "") or "") + " " + (article.get("description", "") or "")
        if text.strip():
            score = analyzer.polarity_scores(text)["compound"]
            scores.append(score)

    return float(np.mean(scores)) if scores else 0.0


def sentiment_to_direction(score):
    """
    Convert sentiment score to market direction and color.
    """
    if score > 0.05:
        return "Positive", "Likely Up 📈", "#22c55e"
    elif score < -0.05:
        return "Negative", "Likely Down 📉", "#ef4444"
    else:
        return "Neutral", "Sideways ⚖️", "#eab308"


def combine_model_sentiment(forecast_df, sentiment_score, sentiment_direction):
    """
    Combine the model forecast trend and sentiment into a final conclusion.
    Always returns a string, never None.
    """
    if forecast_df.empty:
        return f"Forecast unavailable — Sentiment: {sentiment_direction} ⚖️"

    trend = forecast_df["Forecast"].iloc[-1] - forecast_df["Forecast"].iloc[0]

    if trend > 0 and sentiment_score > 0.05:
        return "Forecast & Sentiment both positive — Strong Buy 📈"
    elif trend < 0 and sentiment_score < -0.05:
        return "Forecast & Sentiment both negative — Strong Sell 📉"
    elif trend > 0 and sentiment_score < -0.05:
        return "Forecast up but sentiment negative — Caution ⚖️"
    elif trend < 0 and sentiment_score > 0.05:
        return "Forecast down but sentiment positive — Watch ⚖️"
    else:
        return f"Mixed signals — Sentiment: {sentiment_direction}, Forecast trend unclear ⚖️"
