from config import Config
from model import get_VAR_forecast

import pandas as pd


def transform_to_weekdays(dates: list):
    dates = pd.to_datetime(dates)

    weekdays = []
    date = dates[0]
    for i in range(len(dates)):
        while date.weekday() > 4:
            date += pd.Timedelta(days=1)
        weekdays.append(date.strftime("%Y-%m-%d"))
        date += pd.Timedelta(days=1)

    return weekdays


def get_price_forecast(ticker):
    df_train = pd.read_csv(
        f"{Config.MERGED_DATA_PATH}/{ticker}.csv", parse_dates=["datetime"]
    )
    df_train = df_train.set_index("datetime")[-Config.WINDOW_SIZE :]
    forecast = get_VAR_forecast(df_train, Config.FORECAST_HORIZON)
    forecast_dates = forecast.index.to_list()
    weekdays = transform_to_weekdays(forecast_dates)
    forecast.index = pd.to_datetime(weekdays)
    return forecast
