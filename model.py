import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        ))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        ))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = out[:, :, :x.size(2)]
        res = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + res)


class AttentionPooling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, y):
        h = y.transpose(1, 2)
        weights = torch.softmax(self.score(h), dim=1)
        pooled = torch.sum(weights * h, dim=1)
        return pooled, weights.squeeze(-1)


def _tcn_encoder(input_dim, channels, kernel_size, dropout):
    layers = []
    for i, out_ch in enumerate(channels):
        in_ch = input_dim if i == 0 else channels[i - 1]
        layers.append(
            TemporalBlock(in_ch, out_ch, kernel_size=kernel_size, dilation=2 ** i, dropout=dropout)
        )
    return nn.Sequential(*layers)


class RegimeFiLM(nn.Module):
    def __init__(self, n_regimes, channels):
        super().__init__()
        self.gamma = nn.Embedding(n_regimes, channels)
        self.beta = nn.Embedding(n_regimes, channels)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, y, regime):
        # y: [B, C] or [B, C, L]; regime: [B]
        gamma = self.gamma(regime)
        beta = self.beta(regime)
        if y.dim() == 3:
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
        return gamma * y + beta


class RegimeAwareTCN(nn.Module):

    def __init__(self, input_dim, channels, kernel_size=3, dropout=0.2, n_regimes=3):
        super().__init__()
        self.encoder = _tcn_encoder(input_dim, channels, kernel_size, dropout)
        hidden_dim = channels[-1]

        self.pool = AttentionPooling(hidden_dim)
        self.regime_film = RegimeFiLM(n_regimes, hidden_dim)
        self.forecast_head = nn.Linear(hidden_dim, input_dim)
        self.recon_head = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)
        self.regime_head = nn.Linear(hidden_dim, n_regimes)

    def forward(self, x, regime):
        y = self.encoder(x.transpose(1, 2))        # [B, C, L]
        features, attn = self.pool(y)               # [B, C], [B, L]
        regime_logits = self.regime_head(features) 

        y_cond = self.regime_film(y, regime)         # [B, C, L]
        features_cond, _ = self.pool(y_cond)

        forecast = self.forecast_head(features_cond)    # [B, D]
        recon = self.recon_head(y_cond).transpose(1, 2)  # [B, L, D]

        return {
            "forecast": forecast,
            "reconstruction": recon,
            "features": features_cond,
            "regime_logits": regime_logits,
            "attention": attn,
        }


TCN = RegimeAwareTCN


class PlainTCNForecaster(nn.Module):
    def __init__(self, input_dim, channels, kernel_size=3, dropout=0.2):
        super().__init__()
        self.encoder = _tcn_encoder(input_dim, channels, kernel_size, dropout)
        self.pool = AttentionPooling(channels[-1])
        self.forecast_head = nn.Linear(channels[-1], input_dim)

    def forward(self, x):
        y = self.encoder(x.transpose(1, 2))
        features, _ = self.pool(y)
        return {"forecast": self.forecast_head(features)}


class ConvAutoencoder(nn.Module):
    def __init__(self, input_dim, channels, kernel_size=3, dropout=0.2):
        super().__init__()
        self.encoder = _tcn_encoder(input_dim, channels, kernel_size, dropout)
        self.recon_head = nn.Conv1d(channels[-1], input_dim, kernel_size=1)

    def forward(self, x):
        y = self.encoder(x.transpose(1, 2))
        recon = self.recon_head(y).transpose(1, 2)
        return {"reconstruction": recon}
