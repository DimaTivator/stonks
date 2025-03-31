from config import Config

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta


def display_news(news_df):
    st.subheader("Latest News")
    for _, row in news_df.head(20).iterrows():
        with st.container():
            title = row.get("title", "No Title")
            news_time = row["datetime"].strftime("%Y-%m-%d %H:%M")
            sentiment = row.get("sentiment", 0)

            if sentiment < -0.5:
                sentiment_color = "red"
            elif sentiment > 0.5:
                sentiment_color = "green"
            else:
                sentiment_color = "orange"

            st.markdown(f"##### {title}")

            st.markdown(
                f'Sentiment Score: <span style="color:{sentiment_color}; font-weight: bold;">{sentiment:.2f}</span>',
                unsafe_allow_html=True,
            )

            st.markdown(f"*{news_time}*")

            if "summary" in row and pd.notna(row["summary"]):
                st.write(row["summary"])

            if "link" in row and pd.notna(row["link"]):
                st.markdown(f"[Read more]({row['link']})")

            st.markdown("---")


def display_charts(ticker, stock_df, forecast_df):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.1
    )

    fig.add_trace(
        go.Scatter(
            x=stock_df["datetime"],
            y=stock_df["Close"],
            mode="lines+markers",
            name="Historical",
        ),
        row=1,
        col=1,
    )

    last_hist_value = stock_df["Close"].iloc[-1]
    last_pred_value = forecast_df["Close"].iloc[-1]
    predicted_color = "green" if last_pred_value > last_hist_value else "red"

    fig.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df["Close"],
            mode="lines+markers",
            name="Predicted",
            line=dict(color=predicted_color),
        ),
        row=1,
        col=1,
    )

    if "Volume" in stock_df.columns:
        fig.add_trace(
            go.Bar(
                x=stock_df["datetime"],
                y=stock_df["Volume"],
                name="Volume",
                marker=dict(color="rgba(50, 50, 150, 0.3)"),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title=f"{ticker} Stock Prices and Forecast",
        xaxis2=dict(title="Date"),
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Volume"),
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def get_stock_data(ticker):
    stock_file = os.path.join(Config.STOCKS_DATA_PATH, f"{ticker}.csv")

    try:
        stock_df = pd.read_csv(stock_file)
        stock_df["datetime"] = pd.to_datetime(stock_df["datetime"])
        stock_df.sort_values("datetime", inplace=True)
        return stock_df

    except Exception as e:
        st.error(f"Error loading stock data for {ticker}: {e}")
        return


@st.cache_data
def get_news_data(ticker):
    news_file = os.path.join(Config.NEWS_DATA_PATH, f"{ticker}.csv")

    try:
        news_df = pd.read_csv(news_file)
        news_df["datetime"] = pd.to_datetime(news_df["datetime"])
        news_df = news_df.drop(columns=["ticker"])
        news_df.sort_values("datetime", ascending=False, inplace=True)
        return news_df

    except Exception as e:
        st.error(f"Error loading news data for {ticker}: {e}")
        return


@st.cache_data
def get_forecast_data(ticker):
    forecast_df = pd.read_csv(
        os.path.join(Config.FORECAST_DATA_PATH, f"{ticker}.csv"), index_col=0
    )
    return forecast_df


def main():
    tickers = Config.TICKERS

    st.sidebar.title("Select Ticker")
    ticker = st.sidebar.radio(" ", tickers)

    stock_df = get_stock_data(ticker)
    forecast_df = get_forecast_data(ticker)
    news_df = get_news_data(ticker)

    display_charts(ticker, stock_df, forecast_df)
    display_news(news_df)


if __name__ == "__main__":
    main()
