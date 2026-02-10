import torch
from src.data_prep import (
    fetch_data,
    train_test_split,
    create_sequences,
    scale_series,
    run_arima,
    run_sarima,
    run_prophet
)
from src.deep_models import LSTMModel, GRUModel, TransformerModel
from src.train import train_model


# ===========================
# MODEL SELECTION (INPUT)
# ===========================
def get_model_choice():
    print("\nSelect model to run:")
    print("1 - ARIMA")
    print("2 - SARIMA")
    print("3 - Prophet")
    print("4 - LSTM")
    print("5 - GRU")
    print("6 - Transformer")

    choice = input("\nEnter choice (1-6): ").strip()

    mapping = {
        "1": "arima",
        "2": "sarima",
        "3": "prophet",
        "4": "lstm",
        "5": "gru",
        "6": "transformer"
    }

    if choice not in mapping:
        raise ValueError("Invalid choice. Please select 1–6.")

    return mapping[choice]


def main():
    MODEL_TYPE = get_model_choice()

    # 1. Fetch data
    df = fetch_data("^GSPC", "2010-01-01", "2024-12-31")

    # 2. Train-test split
    train_df, test_df = train_test_split(df, "2022-01-01")

    # ===========================
    # CLASSICAL MODELS
    # ===========================
    if MODEL_TYPE == "arima":
        preds = run_arima(train_df["Close"].values)
        print("\n📈 ARIMA Forecast:")
        print(preds)
        return

    if MODEL_TYPE == "sarima":
        preds = run_sarima(train_df["Close"].values)
        print("\n📈 SARIMA Forecast:")
        print(preds)
        return

    if MODEL_TYPE == "prophet":
        forecast = run_prophet(train_df)
        print("\n📈 Prophet Forecast:")
        print(forecast.tail())
        return

    # ===========================
    # DEEP LEARNING MODELS
    # ===========================
    series = train_df["Close"].values.reshape(-1, 1)
    series_scaled, scaler = scale_series(series)

    X, y = create_sequences(
        series_scaled,
        lookback=60,
        horizon=7
    )

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)

    if MODEL_TYPE == "lstm":
        model = LSTMModel(horizon=7)

    elif MODEL_TYPE == "gru":
        model = GRUModel(horizon=7)

    elif MODEL_TYPE == "transformer":
        model = TransformerModel(horizon=7)

    else:
        raise ValueError("Invalid MODEL_TYPE")

    train_model(model, X, y, epochs=25)


if __name__ == "__main__":
    main()
