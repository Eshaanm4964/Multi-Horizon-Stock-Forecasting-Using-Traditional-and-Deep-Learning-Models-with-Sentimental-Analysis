import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, horizon=7):
        super().__init__()

        # ✅ LSTM layer (ADD IT HERE)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Fully connected layer for multi-horizon output
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)

        # Take last time step
        out = out[:, -1, :]

        return self.fc(out)

import torch
import torch.nn as nn

# ---------------------------
# LSTM (UNCHANGED)
# ---------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, horizon=7):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, horizon=7):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

# ---------------------------
# Transformer (NEW)
# ---------------------------
class TransformerModel(nn.Module):
    def __init__(self, d_model=32, nhead=2, layers=1, horizon=7):
        super().__init__()

        self.embedding = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers
        )

        self.fc = nn.Linear(d_model, horizon)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x[:, -1])
