📈 Multi-Horizon Stock Forecasting with Market Sentiment

A **Streamlit** app for multi-horizon stock price forecasting using **classical (ARIMA, SARIMA, Prophet)** and **deep learning (LSTM, GRU, Transformer)** models, combined with **live market sentiment analysis** from news articles.  

## 🚀 Features

- **Classical & Deep Learning Models**: Forecast stock prices over a selected horizon.  
- **Multi-Horizon Forecasting**: Predict stock trends for 1–30 days.  
- **Live Market Sentiment**: Fetches news articles and calculates sentiment to inform trading decisions.  
- **Combined Insights**: Generates a final recommendation based on both model forecast and sentiment.  
- **Interactive Visualization**: Historical price charts and forecast plots using Plotly.  

## 📊 Models Supported

- **Classical**: ARIMA, SARIMA, Prophet  
- **Deep Learning**: LSTM, GRU, Transformer  

## 🧠 Market Sentiment Analysis

- Fetches recent news articles for a stock using **NewsAPI**.  
- Computes sentiment using **VADER Sentiment Analysis**.  
- Converts sentiment score to expected market direction (Up, Down, Sideways).  
- Combines sentiment with forecast trend for actionable insights.  

## 🛠️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/stock-forecast-app.git
cd stock-forecast-app
Create a virtual environment

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
Install dependencies

pip install -r requirements.txt
Set NewsAPI key
Create a .env file in the root directory:

NEWS_API_KEY=your_newsapi_key_here
⚙️ Usage
Run the Streamlit app:

streamlit run app.py
Configure Model, Ticker, Lookback Window, Forecast Horizon, and Epochs in the sidebar.

Click Run Forecast to generate predictions and sentiment insights.

📈 Output
Historical Price Chart – Visualizes past stock prices.

Forecast Chart – Shows predicted prices over selected horizon.

Market Sentiment – Live news sentiment displayed with expected direction.

Combined Recommendation – Final actionable conclusion: Buy / Sell / Caution / Mixed signals.

🗂️ Project Structure
stock-forecast-app/
│
├─ app.py                  # Main Streamlit application
├─ requirements.txt        # Python dependencies
├─ .env                    # Environment variables (NewsAPI key)
├─ src/
│   ├─ sentiment.py        # Fetch & process news sentiment
│   ├─ data_prep.py        # Data fetching & preprocessing
│   ├─ deep_models.py      # LSTM, GRU, Transformer model definitions
│   └─ train.py            # Training utilities
└─ README.md
🔑 Notes
NewsAPI limits: Free NewsAPI key allows a limited number of requests per day. If news is unavailable, sentiment analysis will show Unavailable.

Ensure ticker symbols are valid (e.g., AAPL, ^GSPC).

🛠️ Dependencies
Python 3.10+

Streamlit

Pandas, NumPy

Plotly

PyTorch

NewsAPI Python client

VaderSentiment

Install via:

pip install streamlit pandas numpy plotly torch newsapi-python vaderSentiment
📌 License
This project is licensed under the MIT License – see the LICENSE file for details.

👤 Author
Eshaan Michael – Data Scientist & AI Engineer
GitHub Profile



