import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlexibleModel(nn.Module):
    def __init__(self, n_features, model_type, hidden_size, num_layers,
                 dropout, d_model=64, nhead=4):
        super().__init__()
        self.model_type = model_type

        if model_type == "lstm":
            self.rnn = nn.LSTM(n_features, hidden_size, num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        elif model_type == "gru":
            self.rnn = nn.GRU(n_features, hidden_size, num_layers,
                              batch_first=True,
                              dropout=dropout if num_layers > 1 else 0)
        elif model_type == "tcn":
            layers, in_ch = [], n_features
            for dilation in [1, 2, 4, 8, 16]:
                layers += [
                    nn.Conv1d(in_ch, hidden_size, 3,
                              padding=(3 - 1) * dilation, dilation=dilation),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
                in_ch = hidden_size
            self.tcn = nn.Sequential(*layers)
        elif model_type == "transformer":
            self.input_proj = nn.Linear(n_features, d_model)
            self.pos_enc    = nn.Parameter(torch.randn(1, 200, d_model) * 0.02)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=hidden_size,
                                           dropout=dropout, batch_first=True),
                num_layers=num_layers,
            )

        out_size   = d_model if model_type == "transformer" else hidden_size
        self.bn    = nn.BatchNorm1d(out_size)
        self.fc1   = nn.Linear(out_size, 48)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(dropout)
        self.fc2   = nn.Linear(48, 1)

    def forward(self, x):
        if self.model_type in ("lstm", "gru"):
            out, _ = self.rnn(x)
            out    = out[:, -1, :]
        elif self.model_type == "tcn":
            out = self.tcn(x.permute(0, 2, 1))
            out = out[:, :, :x.size(1)].mean(dim=2)
        elif self.model_type == "transformer":
            out = self.input_proj(x) + self.pos_enc[:, :x.size(1), :]
            out = self.transformer(out).mean(dim=1)
        out = self.bn(out)
        out = self.drop(self.relu(self.fc1(out)))
        return self.fc2(out).squeeze(-1)
