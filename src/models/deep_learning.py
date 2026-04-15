import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden_size=128, num_layers=2, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 48)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(48, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.bn(out)
        out = self.dropout(self.relu(self.fc1(out)))
        return self.fc2(out).squeeze(-1)


class GRUModel(nn.Module):
    def __init__(self, n_features, hidden_size=128, num_layers=2, dropout=0.4):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 48)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(48, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.bn(out)
        out = self.dropout(self.relu(self.fc1(out)))
        return self.fc2(out).squeeze(-1)


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :x.size(2)]
        return self.dropout(self.relu(self.bn(out)))


class TCNModel(nn.Module):
    def __init__(self, n_features, num_channels=64, kernel_size=3, dropout=0.3):
        super().__init__()
        layers = []
        in_ch = n_features
        for dilation in [1, 2, 4, 8, 16]:
            layers.append(TCNBlock(in_ch, num_channels, kernel_size, dilation, dropout))
            in_ch = num_channels
        self.tcn = nn.Sequential(*layers)
        self.fc1 = nn.Linear(num_channels, 48)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(48, 1)

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.tcn(out)
        out = out.mean(dim=2)
        out = self.dropout(self.relu(self.fc1(out)))
        return self.fc2(out).squeeze(-1)


class TransformerModel(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=3, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 200, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(d_model, 48)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(48, 1)

    def forward(self, x):
        out = self.input_proj(x)
        out = out + self.pos_encoding[:, :x.size(1), :]
        out = self.transformer(out)
        out = out.mean(dim=1)
        out = self.dropout(self.relu(self.fc1(out)))
        return self.fc2(out).squeeze(-1)


def build_lstm(n_features, seq_length=None):
    return LSTMModel(n_features).to(DEVICE)

def build_gru(n_features, seq_length=None):
    return GRUModel(n_features).to(DEVICE)

def build_tcn(n_features, seq_length=None):
    return TCNModel(n_features).to(DEVICE)

def build_transformer(n_features, seq_length=None):
    return TransformerModel(n_features).to(DEVICE)

DL_MODELS = {
    "lstm": build_lstm,
    "gru": build_gru,
    "tcn": build_tcn,
    "transformer": build_transformer,
}



class FlexibleModel(nn.Module):
    def __init__(self, n_features, model_type, hidden_size, num_layers, dropout, d_model=64, nhead=4):
        super().__init__()
        self.model_type = model_type

        if model_type == "lstm":
            self.rnn = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif model_type == "gru":
            self.rnn = nn.GRU(n_features, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif model_type == "tcn":
            layers = []
            in_ch = n_features
            for dilation in [1, 2, 4, 8, 16]:
                padding = (3 - 1) * dilation
                layers.append(nn.Conv1d(in_ch, hidden_size, 3, padding=padding, dilation=dilation))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_ch = hidden_size
            self.tcn = nn.Sequential(*layers)
        elif model_type == "transformer":
            self.input_proj = nn.Linear(n_features, d_model)
            self.pos_enc = nn.Parameter(torch.randn(1, 200, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=hidden_size, dropout=dropout, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        out_size = d_model if model_type == "transformer" else hidden_size
        self.bn = nn.BatchNorm1d(out_size)
        self.fc1 = nn.Linear(out_size, 48)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(48, 1)

    def forward(self, x):
        if self.model_type in ["lstm", "gru"]:
            out, _ = self.rnn(x)
            out = out[:, -1, :]
        elif self.model_type == "tcn":
            out = self.tcn(x.permute(0, 2, 1))
            out = out[:, :, :x.size(1)].mean(dim=2)
        elif self.model_type == "transformer":
            out = self.input_proj(x) + self.pos_enc[:, :x.size(1), :]
            out = self.transformer(out).mean(dim=1)

        out = self.bn(out)
        out = self.drop(self.relu(self.fc1(out)))
        return self.fc2(out).squeeze(-1)
