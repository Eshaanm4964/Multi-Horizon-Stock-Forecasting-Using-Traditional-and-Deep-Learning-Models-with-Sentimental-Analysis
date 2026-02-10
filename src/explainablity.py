import matplotlib.pyplot as plt


def plot_attention(attn_weights):
    plt.imshow(attn_weights, cmap="viridis")
    plt.colorbar()
    plt.title("Transformer Attention Map")
    plt.show()

def compare_trends(prophet_forecast, lstm_preds):
    plt.plot(prophet_forecast["ds"], prophet_forecast["trend"], label="Prophet")
    plt.plot(lstm_preds, label="LSTM")
    plt.legend()
    plt.show()
