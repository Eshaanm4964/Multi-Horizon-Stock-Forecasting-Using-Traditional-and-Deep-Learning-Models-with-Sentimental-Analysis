import yfinance as yf
import pandas as pd
import numpy as np
import time
import os
from sklearn.preprocessing import MinMaxScaler


# ---------------------------
# Fetch data (with cache)
# ---------------------------
def fetch_data(ticker, start, end, retries=3, cache_path="data/cache.csv"):
    for i in range(retries):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False
            )

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.reset_index(inplace=True)

            if len(df) > 0:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                df.to_csv(cache_path, index=False)
                print("✅ Data fetched from Yahoo Finance")
                return df

        except Exception as e:
            print(f"Retry {i+1} failed:", e)
            time.sleep(2)

    if os.path.exists(cache_path):
        print("⚠️ Yahoo unreachable — loading cached data")
        return pd.read_csv(cache_path, parse_dates=["Date"])

    raise RuntimeError("Failed to fetch data and no cache found")


# ---------------------------
# Train-test split (NO lookahead)
# ---------------------------
def train_test_split(df, split_date):
    train = df[df["Date"] < split_date].copy()
    test = df[df["Date"] >= split_date].copy()
    return train, test


# ---------------------------
# Scaling
# ---------------------------
def scale_series(series):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)
    return scaled, scaler


# ---------------------------
# Sequence creation
# ---------------------------
def create_sequences(series, lookback, horizon):
    X, y = [], []

    for i in range(len(series) - lookback - horizon):
        X.append(series[i:i + lookback])
        y.append(series[i + lookback:i + lookback + horizon].flatten())

    return np.array(X), np.array(y)

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet


# ---------------------------
# ARIMA
# ---------------------------
def run_arima(series, horizon=7, steps=None):
    if steps is not None:
        horizon = steps

    model = ARIMA(series, order=(5, 1, 0))
    fitted = model.fit()

    forecast = fitted.forecast(steps=horizon)
    return np.array(forecast)


# ---------------------------
# SARIMA
# ---------------------------
def run_sarima(series, horizon=7, steps=None):
    if steps is not None:
        horizon = steps

    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted = model.fit(disp=False)
    forecast = fitted.forecast(steps=horizon)

    return np.array(forecast)


# ---------------------------
# Prophet
# ---------------------------
def run_prophet(df, horizon=7):
    prophet_df = df[["Date", "Close"]].rename(
        columns={"Date": "ds", "Close": "y"}
    )

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True
    )
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)

    return forecast[["ds", "yhat"]].tail(horizon)

