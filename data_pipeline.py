from config import Config

import scraper
import sentiment
from price_forecast import get_price_forecast

import sys
import os
import pandas as pd


def create_dirs():
    os.makedirs(Config.NEWS_DATA_PATH, exist_ok=True)
    os.makedirs(Config.STOCKS_DATA_PATH, exist_ok=True)
    os.makedirs(Config.WEIGHTS_PATH, exist_ok=True)
    os.makedirs(Config.SENTIMENTS_DATA_PATH, exist_ok=True)
    os.makedirs(Config.MERGED_DATA_PATH, exist_ok=True)
    os.makedirs(Config.FORECAST_DATA_PATH, exist_ok=True)


def merge_stocks_sentiments():
    for ticker in Config.TICKERS:
        stocks_path = os.path.join(Config.STOCKS_DATA_PATH, f"{ticker}.csv")
        sentiments_path = os.path.join(Config.SENTIMENTS_DATA_PATH, f"{ticker}.csv")

        if not os.path.exists(stocks_path) or not os.path.exists(sentiments_path):
            continue

        df_stocks = pd.read_csv(stocks_path)
        df_sentiments = pd.read_csv(sentiments_path).rename(
            columns={"date": "datetime"}
        )

        df_merged = pd.merge(df_stocks, df_sentiments, on="datetime", how="left")

        df_merged = df_merged[["datetime", "Close", "sentiment"]]

        df_merged["sentiment"] = df_merged["sentiment"].fillna(0)

        df_merged.to_csv(
            os.path.join(Config.MERGED_DATA_PATH, f"{ticker}.csv"), index=False
        )


def update_price_sentiment_dataset():
    scraper.update_news(Config.TICKERS, Config.NEWS_DATA_PATH)
    scraper.update_stocks(Config.TICKERS, Config.STOCKS_DATA_PATH)

    sentiment.update_sentiments(
        Config.TICKERS,
        Config.NEWS_DATA_PATH,
        Config.WEIGHTS_PATH,
        Config.SENTIMENT_MODEL_NAME,
    )
    sentiment.update_daily_sentiments(
        Config.TICKERS, Config.NEWS_DATA_PATH, Config.SENTIMENTS_DATA_PATH
    )
    sentiment.fillna_in_daily_sentiments(Config.TICKERS, Config.SENTIMENTS_DATA_PATH)

    merge_stocks_sentiments()


def update_price_forecast():
    for ticker in Config.TICKERS:
        forecast = get_price_forecast(ticker)
        forecast.to_csv(
            os.path.join(Config.FORECAST_DATA_PATH, f"{ticker}.csv"), index=True
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        create_dirs()
    
    update_price_sentiment_dataset()
    update_price_forecast()
