import streamlit as st
import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Multi-Horizon Stock Forecasting",
    layout="wide"
)

# =====================
# IMPORTS
# =====================
from src.sentiment import (
    fetch_stock_news,
    compute_sentiment_score,
    sentiment_to_direction,
    combine_model_sentiment 
)

from src.data_prep import (
    fetch_data,
    train_test_split,
    create_sequences,
    scale_series,
    run_arima,
    run_sarima,
    run_prophet
)

from src.deep_models import (
    LSTMModel,
    GRUModel,
    TransformerModel,
    LSGUModel
)

from src.train import train_model

from src.advanced_models import (
    run_lightgbm,
    run_stacking_ensemble
)

# =====================
# PREMIUM STYLING
# =====================
st.markdown("""
<style>
/* PAGE */
.stApp {
    background: linear-gradient(135deg, #0f1117, #111827);
    color: #e5e7eb;
    font-family: 'Inter', sans-serif;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #1f2937;
}

/* BUTTONS */
.stButton > button {
    background: linear-gradient(135deg, #16a34a, #22c55e);
    color: white;
    font-weight: 600;
    border-radius: 12px;
    height: 3em;
    border: none;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #15803d, #16a34a);
    transform: translateY(-2px);
}

/* CARDS */
.card {
    background: rgba(17,24,39,0.85);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.05);
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    text-align:center;
}

/* HEADINGS */
h1 { color: #22c55e; font-weight: 700; }
h2, h3 { color: #86efac; font-weight: 600; }

/* SENTIMENT BADGES */
.sentiment {
    font-weight: 700;
    padding: 8px 14px;
    border-radius: 12px;
    display: inline-block;
    min-width: 120px;
}
.positive { background: #22c55e33; color: #16a34a; }
.negative { background: #ef444433; color: #b91c1c; }
.neutral { background: #eab30833; color: #ca8a04; }

/* DataFrame hover */
.stDataFrame table tbody tr:hover {
    background-color: rgba(34,197,94,0.1) !important;
}
</style>
""", unsafe_allow_html=True)

# =====================
# HEADER
# =====================
st.title("📈 Multi-Horizon Stock Forecasting")
st.markdown(
    "<span style='color:#9ca3af'>Classical & Deep Learning models with live market sentiment</span>",
    unsafe_allow_html=True
)

# =====================
# SIDEBAR
# =====================
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown("---")

# Model selection
selected_model = st.sidebar.selectbox(
    "🤖 Select Model",
    ["ARIMA", "SARIMA", "Prophet", "LSTM", "GRU", "Transformer", "LSGU", "LightGBM", "Stacking Ensemble"]
)

# Model Info button
if st.sidebar.button("📋 Model Info"):
    # Detailed information for each model
    model_info = {
        "ARIMA": {
            "title": "📊 ARIMA (AutoRegressive Integrated Moving Average)",
            "description": "A classical statistical model that analyzes time series data to identify patterns and make forecasts.",
            "best_for": "Stocks with clear trends and relatively stable patterns",
            "how_works": "Uses past values and forecast errors to predict future prices",
            "pros": "• Interpretable results\n• Works well with limited data\n• Good for short-term forecasts",
            "cons": "• Assumes linear relationships\n• Struggles with volatile markets\n• Requires stationary data"
        },
        "SARIMA": {
            "title": "📈 SARIMA (Seasonal ARIMA)",
            "description": "An extension of ARIMA that captures seasonal patterns in time series data.",
            "best_for": "Stocks with seasonal patterns (retail, agriculture, energy)",
            "how_works": "Adds seasonal components to ARIMA for recurring patterns",
            "pros": "• Captures seasonal trends\n• Handles cyclical patterns\n• Better for seasonal stocks",
            "cons": "• More complex parameters\n• Requires longer data history\n• Slower to train"
        },
        "Prophet": {
            "title": "🔮 Prophet (Facebook's Forecasting Tool)",
            "description": "A modern forecasting tool designed for business time series data.",
            "best_for": "Stocks affected by holidays, events, and business cycles",
            "how_works": "Decomposes time series into trend, seasonality, and holiday effects",
            "pros": "• Automatic holiday detection\n• Handles missing data well\n• Easy to use and interpret",
            "cons": "• Less flexible than neural networks\n• May oversimplify complex patterns\n• Facebook-specific optimizations"
        },
        "LSTM": {
            "title": "🧠 LSTM (Long Short-Term Memory)",
            "description": "A deep learning model capable of learning long-term dependencies in sequential data.",
            "best_for": "Complex stocks with long-term patterns and memory effects",
            "how_works": "Uses memory cells to store information over long sequences",
            "pros": "• Captures complex non-linear patterns\n• Remembers long-term dependencies\n• State-of-the-art for sequences",
            "cons": "• Requires lots of data\n• Computationally expensive\n• Can overfit easily"
        },
        "GRU": {
            "title": "⚡ GRU (Gated Recurrent Unit)",
            "description": "A simplified version of LSTM that's faster to train while maintaining good performance.",
            "best_for": "When you need LSTM-like performance with faster training",
            "how_works": "Uses update and reset gates to control information flow",
            "pros": "• Faster training than LSTM\n• Fewer parameters\n• Good performance on many tasks",
            "cons": "• Less expressive than LSTM\n• May miss very long patterns\n• Simpler architecture"
        },
        "Transformer": {
            "title": "🎯 Transformer",
            "description": "An attention-based neural network that revolutionized sequence modeling.",
            "best_for": "Stocks where specific time periods are more important than others",
            "how_works": "Uses self-attention to weigh importance of different time steps",
            "pros": "• Parallel processing\n• Captures long-range dependencies\n• State-of-the-art performance",
            "cons": "• Requires large datasets\n• Memory intensive\n• Complex to implement"
        },
        "LSGU": {
            "title": "🚀 LSGU (LSTM-GRU Hybrid)",
            "description": "Our custom hybrid model combining LSTM and GRU with attention mechanisms.",
            "best_for": "Maximum accuracy on complex stock patterns",
            "how_works": "LSTM captures long-term memory, GRU handles temporal patterns, attention focuses on important periods",
            "pros": "• Combines strengths of multiple models\n• Attention mechanism for focus\n• Most advanced architecture",
            "cons": "• Most complex model\n• Highest computational cost\n• Requires most data"
        },
        "LightGBM": {
            "title": "💡 LightGBM (Gradient Boosting)",
            "description": "A powerful gradient boosting model that uses technical indicators for forecasting.",
            "best_for": "Stocks where technical analysis indicators are predictive",
            "how_works": "Creates decision trees using features like moving averages, momentum, volatility",
            "pros": "• Very fast training\n• Handles non-linear patterns well\n• Feature importance analysis",
            "cons": "• Limited to provided features\n• May miss sequential patterns\n• Requires feature engineering"
        },
        "Stacking Ensemble": {
            "title": "🎪 Stacking Ensemble",
            "description": "Combines all models to leverage their individual strengths and minimize weaknesses.",
            "best_for": "When you want the most robust and accurate predictions",
            "how_works": "Trains a meta-model to learn optimal weights from all base model predictions",
            "pros": "• Highest accuracy potential\n• Most robust approach\n• Automatic model selection",
            "cons": "• Highest computational cost\n• Complex to debug\n• May be overkill for simple cases"
        }
    }
    
    info = model_info[selected_model]
    
    # Display model info using Streamlit components
    st.sidebar.markdown("---")
    
    # Title
    st.sidebar.markdown(f"### {info['title']}")
    
    # Description
    st.sidebar.info(info['description'])
    
    # Best For
    st.sidebar.markdown("**🎯 Best For:**")
    st.sidebar.write(info['best_for'])
    
    # How It Works
    st.sidebar.markdown("**⚙️ How It Works:**")
    st.sidebar.write(info['how_works'])
    
    # Pros
    st.sidebar.markdown("**✅ Pros:**")
    for pro in info['pros'].split('\n'):
        if pro.strip():
            st.sidebar.write(f"• {pro.strip()}")
    
    # Cons
    st.sidebar.markdown("**❌ Cons:**")
    for con in info['cons'].split('\n'):
        if con.strip():
            st.sidebar.write(f"• {con.strip()}")
    
    st.sidebar.markdown("---")

ticker = st.sidebar.text_input("Ticker", value="^GSPC")
lookback = st.sidebar.slider("Lookback Window", 30, 120, 60)
horizon = st.sidebar.slider("Forecast Horizon (days)", 1, 30, 7)
epochs = st.sidebar.slider("Epochs (DL models)", 5, 30, 15)

run_btn = st.sidebar.button("🚀 Run Forecast")

# =====================
# LIVE MARKET SENTIMENT
# =====================
# =====================
# LIVE MARKET SENTIMENT
# =====================
st.subheader("🧠 Current Market Sentiment")

# Initialize defaults
sentiment_score = 0  # neutral
sentiment = "Unavailable"
direction = "Unknown"
color = "#9ca3af"
articles = []

try:
    articles = fetch_stock_news(ticker)

    if articles:
        sentiment_score = compute_sentiment_score(articles)
        sentiment, direction, color = sentiment_to_direction(sentiment_score)
except:
    pass  # keep defaults


col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="card">
        <h3>Market Sentiment</h3>
        <span class="sentiment {'positive' if sentiment=='Positive' else 'negative' if sentiment=='Negative' else 'neutral'}">{sentiment}</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <h3>Expected Direction</h3>
        <span class="sentiment {'positive' if direction.startswith('Up') else 'negative' if direction.startswith('Down') else 'neutral'}">{direction}</span>
    </div>
    """, unsafe_allow_html=True)

with st.expander("📰 News Driving This Sentiment"):
    if articles:
        for a in articles[:5]:
            st.write("•", a.get("title", "No title"))
    else:
        st.write("No news available right now.")

# =====================
# RUN PIPELINE
# =====================
if run_btn:
    with st.spinner("Fetching data & running model..."):

        # ---------- DATA ----------
        end_date = datetime.now().strftime('%Y-%m-%d')
        df = fetch_data(ticker, "2010-01-01", end_date)
        train_df, test_df = train_test_split(df, "2022-01-01")

        # ---------- HISTORICAL CHART ----------
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Close"],
                mode="lines",
                line=dict(color="#60a5fa", width=2)
            )
        )

        fig_hist.update_layout(
            template="plotly_dark",
            height=450,
            xaxis_title="Date",
            yaxis_title="Price",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.subheader("📊 Historical Price")
        st.plotly_chart(fig_hist, use_container_width=True)

        # =====================
        # CLASSICAL MODELS
        # =====================
        if selected_model in ["ARIMA", "SARIMA"]:

            if selected_model == "ARIMA":
                preds = run_arima(train_df["Close"].values, steps=horizon)
            else:
                preds = run_sarima(train_df["Close"].values, steps=horizon)

            forecast_df = pd.DataFrame({
                "Date": [datetime.now() + timedelta(days=i) for i in range(1, horizon + 1)],
                "Forecast": preds
            })

        elif selected_model == "Prophet":
            forecast = run_prophet(train_df, horizon=horizon)
            forecast_df = forecast[["ds", "yhat"]].tail(horizon)
            forecast_df.columns = ["Date", "Forecast"]

        elif selected_model == "LightGBM":
            preds = run_lightgbm(train_df["Close"].values, horizon=horizon, lookback=lookback)
            forecast_df = pd.DataFrame({
                "Date": [datetime.now() + timedelta(days=i) for i in range(1, horizon + 1)],
                "Forecast": preds
            })

        elif selected_model == "Stacking Ensemble":
            preds = run_stacking_ensemble(train_df["Close"].values, horizon=horizon, lookback=lookback)
            forecast_df = pd.DataFrame({
                "Date": [datetime.now() + timedelta(days=i) for i in range(1, horizon + 1)],
                "Forecast": preds
            })

        # =====================
        # DEEP LEARNING MODELS
        # =====================
        else:
            series = train_df["Close"].values.reshape(-1, 1)
            series_scaled, scaler = scale_series(series)

            X, y = create_sequences(series_scaled, lookback, horizon)

            if selected_model == "Transformer":
                X, y = X[-1000:], y[-1000:]  # RAM safety

            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

            if selected_model == "LSTM":
                model = LSTMModel(horizon=horizon)
            elif selected_model == "GRU":
                model = GRUModel(horizon=horizon)
            elif selected_model == "LSGU":
                model = LSGUModel(horizon=horizon)
            else:
                model = TransformerModel(horizon=horizon)

            train_model(model, X, y, epochs=epochs, batch_size=32)

            with torch.no_grad():
                preds = model(X[-1:]).numpy().flatten()

            preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()

            forecast_df = pd.DataFrame({
                "Date": [datetime.now() + timedelta(days=i) for i in range(1, horizon + 1)],
                "Forecast": preds
            })

        # ---------- FORECAST CHART ----------
        fig_forecast = go.Figure()
        fig_forecast.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Forecast"],
                mode="lines+markers",
                line=dict(color="#22c55e", width=3, shape='spline'),
                marker=dict(size=8)
            )
        )
        # Shaded confidence area
        fig_forecast.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Forecast"]*1.02,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            )
        )
        fig_forecast.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Forecast"]*0.98,
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(34,197,94,0.1)',
                line=dict(width=0),
                showlegend=False
            )
        )

        fig_forecast.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Forecast Horizon",
            yaxis_title="Predicted Price",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.subheader(f"🔮 {selected_model} Forecast — Next {horizon} Days")
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.success("Forecast completed successfully ✅")

        # ---------- COMBINE FORECAST + SENTIMENT ----------
        final_conclusion = combine_model_sentiment(forecast_df, sentiment_score, sentiment)
        st.subheader("📌 Combined Forecast & Sentiment")
        st.markdown(f"""
        <div class="card">
            <h2 style="color:#facc15">{final_conclusion}</h2>
        </div>
        """, unsafe_allow_html=True)
