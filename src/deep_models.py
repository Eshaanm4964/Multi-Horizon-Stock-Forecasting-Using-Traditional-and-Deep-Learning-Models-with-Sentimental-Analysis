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

# ---------------------------
# LSTM-GRU Hybrid (LSGU)
# ---------------------------
class LSGUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, horizon=7):
        super().__init__()
        
        # LSTM layer for long-term dependencies
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=0.2
        )
        
        # GRU layer for capturing temporal patterns
        self.gru = nn.GRU(
            input_size=hidden_size,  # Input from LSTM
            hidden_size=hidden_size,
            batch_first=True,
            dropout=0.2
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, horizon)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        
        # GRU forward pass (takes LSTM output as input)
        gru_out, _ = self.gru(lstm_out)  # (batch, seq_len, hidden_size)
        
        # Apply attention mechanism
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        
        # Residual connection and layer normalization
        combined = self.layer_norm(gru_out + attn_out)
        
        # Take last time step
        last_step = combined[:, -1, :]  # (batch, hidden_size)
        
        # Apply dropout
        last_step = self.dropout(last_step)
        
        # Fully connected layers
        out = self.fc1(last_step)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
