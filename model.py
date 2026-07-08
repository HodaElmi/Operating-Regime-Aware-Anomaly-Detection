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

        out = out[:, :, :x.size(2)]  # remove right padding
        res = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + res)


class AttentionPooling(nn.Module):
    """Small attention pooling layer over the temporal axis."""
    def __init__(self, channels):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, y):
        # y: [B, C, L] -> h: [B, L, C]
        h = y.transpose(1, 2)
        weights = torch.softmax(self.score(h), dim=1)
        pooled = torch.sum(weights * h, dim=1)
        return pooled, weights.squeeze(-1)


class RegimeAwareTCN(nn.Module):
    """TCN encoder with three simple heads.

    Heads:
      forecast_head: predicts the next sensor vector.
      recon_head: reconstructs the input window from temporal features.
      regime_head: predicts the operating regime, used only as an auxiliary loss.

    The architecture stays intentionally lightweight and easy to follow.
    """
    def __init__(self, input_dim, channels, kernel_size=3, dropout=0.2, n_regimes=3):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(channels):
            in_ch = input_dim if i == 0 else channels[i - 1]
            layers.append(
                TemporalBlock(
                    in_ch, out_ch,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout,
                )
            )

        self.encoder = nn.Sequential(*layers)
        hidden_dim = channels[-1]

        self.pool = AttentionPooling(hidden_dim)
        self.forecast_head = nn.Linear(hidden_dim, input_dim)
        self.recon_head = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)
        self.regime_head = nn.Linear(hidden_dim, n_regimes)

    def forward(self, x):
        # x: [B, L, D]
        y = self.encoder(x.transpose(1, 2))       # [B, C, L]
        features, attn = self.pool(y)             # [B, C], [B, L]

        forecast = self.forecast_head(features)   # [B, D]
        recon = self.recon_head(y).transpose(1, 2) # [B, L, D]
        regime_logits = self.regime_head(features)

        return {
            "forecast": forecast,
            "reconstruction": recon,
            "features": features,
            "regime_logits": regime_logits,
            "attention": attn,
        }


# Backwards-compatible alias for older scripts/checkpoints.
TCN = RegimeAwareTCN
