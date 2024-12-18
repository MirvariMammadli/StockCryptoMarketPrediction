import os
import praw
import logging
import requests
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

class DataFetcher:
    def __init__(self, config):
        self.reddit = praw.Reddit(
            client_id=config.reddit_client_id,
            client_secret=config.reddit_client_secret,
            user_agent=config.reddit_user_agent
        )
        self.scaler = MinMaxScaler()

    def fetch_reddit_headlines(self, subreddit_name, limit, save_path):
        print(f"Fetching {limit} headlines from r/{subreddit_name}...")
        subreddit = self.reddit.subreddit(subreddit_name)
        headlines = []
        for post in subreddit.hot(limit=limit):
            headlines.append({'title': post.title, 'score': post.score})

        df = pd.DataFrame(headlines)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df.to_csv(f"{save_path}/{subreddit_name}_headlines.csv", index=False)
        print(f"Headlines saved at {save_path}/{subreddit_name}_headlines.csv")

    def fetch_crypto_data(self, symbol, interval, start_date, end_date, save_path):
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(pd.Timestamp(start_date).timestamp() * 1000),
            "endTime": int(pd.Timestamp(end_date).timestamp() * 1000),
            "limit": 1000
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.rename(columns={"timestamp": "Date"}, inplace=True)
        df.set_index("Date", inplace=True)
        df.to_csv(save_path)
        logging.info(f"Crypto data for {symbol} saved at {save_path}")

    def fetch_stock_data(self, ticker, start_date, end_date, save_path):
        data = yf.download(ticker, start=start_date, end=end_date)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        data.to_csv(save_path)
        logging.info(f"Stock data for {ticker} saved at {save_path}")
