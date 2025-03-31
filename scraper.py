from config import Config

import os
import time
import requests
import datetime
import tempfile
import yfinance as yf
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup


def get_latest_news(ticker: str):
    url = f"https://finviz.com/quote.ashx?t={ticker}&p=d"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch latest news for {ticker}. Status code {response.status_code}"
        )

    html = BeautifulSoup(response.text, features="html.parser")
    finviz_news_table = html.find(id="news-table")

    news_parsed = []
    last_full_date = None

    for row in finviz_news_table.find_all("tr"):
        try:
            headline = row.a.getText()

            date_cell = row.find("td", align="right")
            date_text = date_cell.get_text(strip=True)

            if "-" in date_text:
                last_full_date = date_text
                timestamp = datetime.datetime.strptime(date_text, "%b-%d-%y %I:%M%p")
            else:
                if last_full_date:
                    full_date_str = last_full_date.split()[0] + " " + date_text
                    timestamp = datetime.datetime.strptime(
                        full_date_str, "%b-%d-%y %I:%M%p"
                    )
                else:
                    continue

            news_parsed.append([ticker, timestamp, headline])
        except Exception as e:
            pass

    result = pd.DataFrame(news_parsed, columns=["ticker", "datetime", "title"])
    result["sentiment"] = None
    return result


def update_news(tickers: list[str], data_directory: str) -> None:
    for ticker in tqdm(tickers):
        file_path = os.path.join(data_directory, f"{ticker}.csv")

        news_df = get_latest_news(ticker)

        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            combined_df = pd.concat([existing_df, news_df]).drop_duplicates(
                subset=["title"], keep="last"
            )
        else:
            combined_df = news_df
        combined_df.to_csv(file_path, index=False)


def update_stocks(tickers: list[str], data_directory: str) -> None:
    end_date = datetime.date.today() - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=Config.WINDOW_SIZE)
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as temp_file:
            temp_path = temp_file.name
            data.to_csv(temp_path)

        df = pd.read_csv(temp_path)[2:]
        df = df.rename(columns={"Price": "datetime"}).reset_index(drop=True)

        df.to_csv(os.path.join(data_directory, f"{ticker}.csv"), index=False)

        time.sleep(1)

    os.remove(temp_path)


if __name__ == "__main__":
    update_stocks(Config.TICKERS, Config.STOCKS_DATA_PATH)
