import streamlit as st
import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from google import genai

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Multi-Horizon Stock Forecasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================
# GEMINI SETUP
# =====================
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None


def ask_gemini(user_prompt, context=None):
    if client is None:
        return "⚠️ Gemini API key not found. Check your .env file."
    full_prompt = f"""
You are a professional quantitative financial AI assistant with deep expertise in stock markets,
technical analysis, forecasting models, and investment strategy.
Provide sharp, analytical, objective answers. Be direct and specific.
Avoid generic disclaimers. Format responses clearly.

===== CURRENT CONTEXT =====
{context if context else "No forecast has been run yet."}

===== USER QUESTION =====
{user_prompt}
    """
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=full_prompt,
        )
        return response.text
    except Exception as e:
        return f"Gemini Error: {str(e)}"


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
from src.deep_models import LSTMModel, GRUModel, TransformerModel, LSGUModel
from src.train import train_model
from src.advanced_models import run_lightgbm, run_stacking_ensemble

# =====================
# SESSION STATE
# =====================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "page" not in st.session_state:
    st.session_state.page = "forecast"
if "forecast_context" not in st.session_state:
    st.session_state.forecast_context = None

# =====================
# SHARED STYLES
# =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: #080c12;
    color: #c9d1e0;
    font-family: 'DM Sans', sans-serif;
}
section[data-testid="stSidebar"] {
    background: #0c1018 !important;
    border-right: 1px solid #1a2235 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.05em !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.55rem 1.2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 12px rgba(14,165,233,0.2) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(14,165,233,0.4) !important;
}
h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2rem !important;
    color: #f1f5f9 !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 0.2rem !important;
}
h2, h3 { font-family: 'Syne', sans-serif !important; color: #e2e8f0 !important; }
.metric-card {
    background: #0d1520;
    border: 1px solid #1a2740;
    border-radius: 14px;
    padding: 22px 24px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #0ea5e9, #6366f1);
}
.metric-card:hover { border-color: #2a3f5f; }
.metric-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 12px;
    font-family: 'DM Sans', sans-serif;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9;
}
.badge {
    display: inline-flex;
    align-items: center;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.04em;
}
.badge-positive { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.badge-negative { background: #2d0a0a; color: #f87171; border: 1px solid #7f1d1d; }
.badge-neutral   { background: #1c1a06; color: #fbbf24; border: 1px solid #78350f; }
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #475569;
    margin: 32px 0 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1a2235;
}
.app-header { display: flex; align-items: center; gap: 14px; margin-bottom: 4px; }
.app-header-icon {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; flex-shrink: 0;
}
.app-subtitle {
    font-size: 0.82rem;
    color: #475569;
    font-family: 'DM Sans', sans-serif;
    margin-bottom: 28px;
}
#MainMenu, footer, header { visibility: hidden; }

/* Active nav button */
.nav-active > button {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    box-shadow: 0 2px 16px rgba(16,185,129,0.4) !important;
}
</style>
""", unsafe_allow_html=True)


# =====================
# SIDEBAR — always visible
# =====================
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown("---")

selected_model = st.sidebar.selectbox(
    "Model",
    ["ARIMA", "SARIMA", "Prophet", "LSTM", "GRU", "Transformer", "LSGU", "LightGBM", "Stacking Ensemble"]
)

if st.sidebar.button("📋 Model Info"):
    model_info = {
        "ARIMA":             ("Classical statistical model using past values & errors.",         "Clear trends & short-term forecasts"),
        "SARIMA":            ("ARIMA extended with seasonal components.",                        "Seasonal stocks — retail, energy, agriculture"),
        "Prophet":           ("Decomposition: trend + seasonality + holiday effects.",           "Stocks driven by business cycles & holidays"),
        "LSTM":              ("Deep learning with long-term memory cells.",                      "Complex stocks with extended pattern memory"),
        "GRU":               ("Faster LSTM variant with gating mechanisms.",                     "LSTM performance with faster training"),
        "Transformer":       ("Attention-based — weights time steps by importance.",             "When specific periods dominate the signal"),
        "LSGU":              ("Custom LSTM + GRU hybrid with attention.",                        "Maximum accuracy on complex patterns"),
        "LightGBM":          ("Gradient boosting over technical indicators.",                    "Technically driven stocks"),
        "Stacking Ensemble": ("Meta-model combining all base model predictions.",                "Highest robustness & accuracy needs"),
    }
    desc, best = model_info[selected_model]
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{selected_model}**")
    st.sidebar.info(desc)
    st.sidebar.caption(f"**Best for:** {best}")
    st.sidebar.markdown("---")

ticker   = st.sidebar.text_input("Ticker Symbol", value="^GSPC")
lookback = st.sidebar.slider("Lookback Window", 30, 120, 60)
horizon  = st.sidebar.slider("Forecast Horizon (days)", 1, 30, 7)
epochs   = st.sidebar.slider("Training Epochs", 5, 30, 15)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Run Forecast")

st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Chat button — switches to chat view
chat_btn_container = st.sidebar.container()
with chat_btn_container:
    st.markdown("""
    <style>
    div[data-testid="stSidebar"] div.chat-nav-btn > div > button {
        background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
        box-shadow: 0 2px 16px rgba(99,102,241,0.35) !important;
        font-size: 0.85rem !important;
        padding: 0.7rem 1.2rem !important;
    }
    div[data-testid="stSidebar"] div.chat-nav-btn > div > button:hover {
        box-shadow: 0 4px 24px rgba(99,102,241,0.55) !important;
    }
    </style>
    <div class="chat-nav-btn">
    """, unsafe_allow_html=True)
    if st.button("💬 Chat with Financial AI"):
        st.session_state.page = "chat"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.page == "chat":
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <style>
    div[data-testid="stSidebar"] div.back-btn > div > button {
        background: #0d1520 !important;
        border: 1px solid #1a2740 !important;
        color: #94a3b8 !important;
        box-shadow: none !important;
        font-size: 0.78rem !important;
    }
    </style>
    <div class="back-btn">
    """, unsafe_allow_html=True)
    if st.sidebar.button("← Back to Forecast"):
        st.session_state.page = "forecast"
        st.rerun()
    st.sidebar.markdown("</div>", unsafe_allow_html=True)


# ================================================
# PAGE: FORECAST
# ================================================
if st.session_state.page == "forecast":

    # Header
    st.markdown("""
    <div class="app-header">
        <div class="app-header-icon">📈</div>
        <h1>Multi-Horizon Stock Forecasting</h1>
    </div>
    <p class="app-subtitle">Classical &amp; deep learning models &nbsp;·&nbsp; Live market sentiment &nbsp;·&nbsp; AI-powered analysis</p>
    """, unsafe_allow_html=True)

    # Sentiment
    sentiment_score = 0
    sentiment = "Unavailable"
    direction = "Unknown"
    articles  = []
    try:
        articles = fetch_stock_news(ticker)
        if articles:
            sentiment_score = compute_sentiment_score(articles)
            sentiment, direction, _ = sentiment_to_direction(sentiment_score)
    except:
        pass

    def badge_cls(v):
        v = str(v)
        if v == "Positive" or v.startswith("Up"):   return "badge-positive"
        if v == "Negative" or v.startswith("Down"): return "badge-negative"
        return "badge-neutral"

    st.markdown('<div class="section-header">Market Overview</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Market Sentiment</div>
            <div class="metric-value"><span class="badge {badge_cls(sentiment)}">{sentiment}</span></div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Expected Direction</div>
            <div class="metric-value"><span class="badge {badge_cls(direction)}">{direction}</span></div>
        </div>""", unsafe_allow_html=True)
    with c3:
        sc = "#4ade80" if sentiment_score > 0 else "#f87171" if sentiment_score < 0 else "#fbbf24"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Sentiment Score</div>
            <div class="metric-value" style="color:{sc};font-size:2.2rem;font-weight:800">{round(sentiment_score,2):+.2f}</div>
        </div>""", unsafe_allow_html=True)

    with st.expander("📰 News Headlines Driving This Sentiment"):
        if articles:
            for a in articles[:5]:
                st.markdown(f"<span style='color:#475569;font-size:0.75rem;margin-right:6px'>▸</span>{a.get('title','No title')}", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#475569;font-size:0.82rem'>No news available right now.</span>", unsafe_allow_html=True)

    # Forecast
    st.markdown('<div class="section-header">Forecast</div>', unsafe_allow_html=True)

    if run_btn:
        with st.spinner("Fetching data & training model…"):
            end_date = datetime.now().strftime('%Y-%m-%d')
            df = fetch_data(ticker, "2010-01-01", end_date)
            train_df, test_df = train_test_split(df, "2022-01-01")

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=df["Date"], y=df["Close"],
                mode="lines", line=dict(color="#0ea5e9", width=1.8),
                fill='tozeroy', fillcolor='rgba(14,165,233,0.05)'
            ))
            fig_hist.update_layout(
                template="plotly_dark", height=300,
                margin=dict(l=0, r=0, t=8, b=0),
                xaxis_title="Date", yaxis_title="Price",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='DM Sans', color='#64748b'),
                xaxis=dict(gridcolor='#0d1520'), yaxis=dict(gridcolor='#0d1520'),
            )
            st.markdown("**Historical Price**")
            st.plotly_chart(fig_hist, use_container_width=True)

            if selected_model in ["ARIMA", "SARIMA"]:
                preds = run_arima(train_df["Close"].values, steps=horizon) if selected_model == "ARIMA" \
                        else run_sarima(train_df["Close"].values, steps=horizon)
                forecast_df = pd.DataFrame({
                    "Date": [datetime.now() + timedelta(days=i) for i in range(1, horizon+1)],
                    "Forecast": preds
                })
            elif selected_model == "Prophet":
                forecast = run_prophet(train_df, horizon=horizon)
                forecast_df = forecast[["ds","yhat"]].tail(horizon).rename(columns={"ds":"Date","yhat":"Forecast"})
            elif selected_model == "LightGBM":
                preds = run_lightgbm(train_df["Close"].values, horizon=horizon, lookback=lookback)
                forecast_df = pd.DataFrame({
                    "Date": [datetime.now() + timedelta(days=i) for i in range(1, horizon+1)],
                    "Forecast": preds
                })
            elif selected_model == "Stacking Ensemble":
                preds = run_stacking_ensemble(train_df["Close"].values, horizon=horizon, lookback=lookback)
                forecast_df = pd.DataFrame({
                    "Date": [datetime.now() + timedelta(days=i) for i in range(1, horizon+1)],
                    "Forecast": preds
                })
            else:
                series = train_df["Close"].values.reshape(-1, 1)
                series_scaled, scaler = scale_series(series)
                X, y = create_sequences(series_scaled, lookback, horizon)
                if selected_model == "Transformer":
                    X, y = X[-1000:], y[-1000:]
                X = torch.tensor(X, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)
                model_map = {"LSTM": LSTMModel, "GRU": GRUModel, "LSGU": LSGUModel, "Transformer": TransformerModel}
                model = model_map[selected_model](horizon=horizon)
                train_model(model, X, y, epochs=epochs, batch_size=32)
                with torch.no_grad():
                    preds = model(X[-1:]).numpy().flatten()
                preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
                forecast_df = pd.DataFrame({
                    "Date": [datetime.now() + timedelta(days=i) for i in range(1, horizon+1)],
                    "Forecast": preds
                })

            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=forecast_df["Date"], y=forecast_df["Forecast"],
                mode="lines+markers",
                line=dict(color="#10b981", width=2.5),
                marker=dict(size=7, color="#10b981", line=dict(color="#fff", width=1.5)),
                fill='tozeroy', fillcolor='rgba(16,185,129,0.06)'
            ))
            fig_fc.update_layout(
                template="plotly_dark", height=280,
                margin=dict(l=0, r=0, t=8, b=0),
                xaxis_title="Date", yaxis_title="Predicted Price",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='DM Sans', color='#64748b'),
                xaxis=dict(gridcolor='#0d1520'), yaxis=dict(gridcolor='#0d1520'),
            )
            st.markdown(f"**{selected_model} Forecast — Next {horizon} Days**")
            st.plotly_chart(fig_fc, use_container_width=True)

            final_conclusion = combine_model_sentiment(forecast_df, sentiment_score, sentiment)

            # Save context for AI chat
            st.session_state.forecast_context = (
                f"Ticker: {ticker} | Model: {selected_model} | "
                f"Horizon: {horizon} days | Sentiment: {sentiment} ({round(sentiment_score,2)}) | "
                f"Direction: {direction} | Verdict: {final_conclusion}\n\n"
                f"Forecast prices:\n{forecast_df.to_string(index=False)}"
            )

            st.markdown('<div class="section-header">Combined Signal</div>', unsafe_allow_html=True)
            vc = "#4ade80" if any(k in final_conclusion for k in ("Bull","Up","Positive","Buy")) \
                 else "#f87171" if any(k in final_conclusion for k in ("Bear","Down","Negative","Sell")) \
                 else "#fbbf24"
            st.markdown(f"""
            <div class="metric-card" style="padding:24px 28px">
                <div class="metric-label">Forecast + Sentiment Verdict</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:700;color:{vc};margin-top:10px">{final_conclusion}</div>
            </div>""", unsafe_allow_html=True)
            st.success("✅ Forecast completed successfully")

    else:
        st.markdown("""
        <div class="metric-card" style="text-align:center;padding:64px 24px;border-style:dashed;border-color:#1a2235">
            <div style="font-size:3rem;margin-bottom:16px;opacity:0.35">📊</div>
            <div style="font-family:'Syne',sans-serif;font-size:1rem;color:#475569;font-weight:600">Select a model in the sidebar and click <em>Run Forecast</em></div>
            <div style="font-size:0.78rem;color:#334155;margin-top:8px">ARIMA · SARIMA · Prophet · LSTM · GRU · Transformer · LSGU · LightGBM · Stacking Ensemble</div>
        </div>""", unsafe_allow_html=True)


elif st.session_state.page == "chat":
    # Custom CSS for the Chat Interface (Injecting into the main app)
    st.markdown("""
    <style>
    /* Chat Container Tweaks */
    .stChatMessage {
        background-color: #0d1520 !important;
        border: 1px solid #1a2740 !important;
        border-radius: 12px !important;
        margin-bottom: 1rem !important;
    }
    .stChatFloatingInputContainer {
        background-color: #080c12 !important;
    }
    .ctx-card {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        font-size: 0.85rem;
        color: #94a3b8;
    }
    .ctx-label {
        color: #0ea5e9;
        font-weight: bold;
        text-transform: uppercase;
        font-size: 0.7rem;
        margin-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- CHAT HEADER ---
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
        <div style="width: 45px; height: 45px; background: #6366f1; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 24px;">🤖</div>
        <div>
            <h2 style="margin:0; font-size: 1.5rem;">Financial AI Assistant</h2>
            <p style="margin:0; color: #64748b; font-size: 0.8rem;">Powered by Gemini · Context-Aware Analysis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- SIDEBAR/CONTEXT DISPLAY ---
    if st.session_state.forecast_context:
        with st.expander("📊 View Current Forecast Context", expanded=False):
            st.markdown(f"""<div class="ctx-card">{st.session_state.forecast_context.replace('|', '<br>')}</div>""", unsafe_allow_html=True)

    # --- DISPLAY CHAT HISTORY ---
    for role, msg in st.session_state.chat_history:
        avatar = "👤" if role == "You" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg)

    # --- CHAT INPUT ---
    if prompt := st.chat_input("Ask about the forecast, risk, or market sentiment..."):
        # Add user message to history
        st.session_state.chat_history.append(("You", prompt))
        
        # Display user message immediately
        with st.chat_message("You", avatar="👤"):
            st.markdown(prompt)

        # Generate AI response
        with st.chat_message("AI", avatar="🤖"):
            with st.spinner("Analyzing market data..."):
                ctx = st.session_state.forecast_context or "No forecast has been run yet."
                answer = ask_gemini(prompt, ctx)
                st.markdown(answer)
        
        # Add AI response to history and rerun to lock UI
        st.session_state.chat_history.append(("AI", answer))
        st.rerun()