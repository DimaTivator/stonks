from config import Config

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

import pandas as pd
from statsmodels.tsa.api import VAR


device = Config.DEVICE


# ---------------------------------------------- FinBERT ----------------------------------------------


class FinBert(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_config(config)

        self.model.classifier = torch.nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(768, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(512, 5),
        )

    def forward(self, *args, **kwargs):
        x = self.model(*args, **kwargs).logits
        x = self.fc(x)
        return x


def get_model(weights_path):
    model = FinBert()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    return model, tokenizer


# ---------------------------------------------- VAR ----------------------------------------------


def get_VAR_forecast(df_train, forecast_horizon):
    model = VAR(df_train)
    lags = model.select_order(maxlags=7)
    optimal_lag = lags.aic

    var_model = model.fit(optimal_lag, trend="c")
    last_values = df_train.values[-optimal_lag:]
    forecast = var_model.forecast(y=last_values, steps=forecast_horizon)

    forecast_df = pd.DataFrame(
        forecast,
        columns=df_train.columns,
        index=pd.date_range(
            start=df_train.index[-1], periods=forecast_horizon + 1, freq="D"
        )[1:],
    )

    return forecast_df
