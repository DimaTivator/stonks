import torch


class Config:
    DEVICE = torch.device("cpu")
    SENTIMENT_MODEL_NAME = "finbert"

    NEWS_DATA_PATH = "./data/news"
    WEIGHTS_PATH = "./weights"
    STOCKS_DATA_PATH = "./data/stocks"
    SENTIMENTS_DATA_PATH = "./data/sentiments"
    MERGED_DATA_PATH = "./data/merged"
    FORECAST_DATA_PATH = "./data/forecast"

    FORECAST_HORIZON = 15
    WINDOW_SIZE = 60

    TICKERS = ["ADI", "AEP", "AMD", "AMZN", "ASML", "AMAT", "AAPL"]
