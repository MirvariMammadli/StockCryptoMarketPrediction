# Initializing Libraries
import os
import gym
import praw
import logging
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from pgmpy.models import BayesianNetwork
from pgmpy.inference import BeliefPropagation
from pgmpy.estimators import MaximumLikelihoodEstimator

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("vader_lexicon")

# Creating class
class FinancialDataPipeline:
    # Define initial variables
    def __init__(self, config):
        self.config = config
        self.reddit = praw.Reddit(
            client_id=config["reddit_client_id"],
            client_secret=config["reddit_client_secret"],
            user_agent=config["reddit_user_agent"]
        )
        self.sia = SentimentIntensityAnalyzer()
        self.scaler = MinMaxScaler()
        logging.basicConfig(level=logging.INFO)

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

    @staticmethod
    def preprocess_headlines(headline):
        tokens = word_tokenize(headline)
        tokens = [word.lower() for word in tokens if word.isalnum()]
        tokens = [word for word in tokens if word not in stopwords.words("english")]
        return " ".join(tokens)

    def analyze_sentiment(self, data_path, save_path):
        reddit_data = pd.read_csv(data_path)
        reddit_data["processed_title"] = reddit_data["title"].apply(self.preprocess_headlines)
        
        # Compute sentiment scores
        reddit_data["sentiment_score"] = reddit_data["processed_title"].apply(
            lambda x: self.sia.polarity_scores(x)["compound"]
        )
        
        reddit_data.to_csv(save_path, index=False)
        print("Processed sentiment data saved at:", save_path)
        
    # Fetch crypto data from Binance using requests
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

    def add_technical_indicators(self, data):
        # Normalize column names to lowercase
        data.columns = [col.lower() for col in data.columns]

        # Add technical indicators
        data["sma_20"] = data["close"].rolling(window=20).mean()
        data["ema_20"] = data["close"].ewm(span=20, adjust=False).mean()
        data["rsi_14"] = self.calculate_rsi(data["close"])
        short_ema = data["close"].ewm(span=12, adjust=False).mean()
        long_ema = data["close"].ewm(span=26, adjust=False).mean()
        data["macd"] = short_ema - long_ema
        data["signal_line"] = data["macd"].ewm(span=9, adjust=False).mean()
        return data

    @staticmethod
    def calculate_rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def lstm_prediction(self, data, look_back=60):
        data_scaled = self.scaler.fit_transform(data.values.reshape(-1, 1))
        X, y = [], []
        for i in range(look_back, len(data_scaled)):
            X.append(data_scaled[i - look_back:i, 0])
            y.append(data_scaled[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Splitting data into test and train parts 
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

        predictions = model.predict(X_test)
        predictions = self.scaler.inverse_transform(predictions)
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))

        plt.figure(figsize=(10, 5))
        plt.plot(y_test_actual, label="Actual")
        plt.plot(predictions, label="Predicted")
        plt.title("LSTM Prediction vs. Actual")
        plt.xlabel("Time Steps")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.show()

        return model, predictions

    def build_bayesian_network(self, data):
        # Discretize continuous data into bins
        discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
        data_discrete = data.copy()

        # Handle missing values
        data_discrete.fillna(method="ffill", inplace=True)
        data_discrete.fillna(method="bfill", inplace=True)
        data_discrete.fillna(0, inplace=True)  # Final fallback for NaNs

        # Validate columns before discretization and discretize
        columns_to_discretize = ["rsi_14", "close", "macd"]
        for col in columns_to_discretize:
            if data_discrete[col].isnull().any():
                raise ValueError(f"NaN values found in column {col} after preprocessing.")
            data_discrete[col] = discretizer.fit_transform(data_discrete[[col]]).astype(int)

        # Define the Bayesian Network structure
        model = BayesianNetwork([("rsi_14", "close"), ("macd", "close")])
        model.fit(data_discrete, estimator=MaximumLikelihoodEstimator)

        # Normalize and validate CPDs
        for cpd in model.get_cpds():
            cpd.normalize()  # Ensures sum of probabilities equals 1
            if not np.isclose(cpd.values.sum(axis=0), 1, atol=0.01).all():
                logging.error(f"CPD for {cpd.variable} still not normalized properly.")
                return None  # or handle differently based on your requirements

        # Perform inference using Belief Propagation
        inference = BeliefPropagation(model)
        try:
            query_result = inference.query(variables=["close"], evidence={"rsi_14": 2}, show_progress=False)
        except Exception as e:
            logging.error(f"Error during inference: {e}")
            return None
        logging.info(f"Bayesian Network Query Result: {query_result}")
        return query_result

    
    def reinforcement_learning(self, env_name="CartPole-v1", episodes=100):
        env = gym.make(env_name)
        rewards = []
        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = env.action_space.sample()
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            rewards.append(total_reward)
        env.close()
        # Plot rewards
        plt.plot(range(episodes), rewards)
        plt.title(f"Rewards Over {episodes} Episodes in {env_name}")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.show()
        return rewards

    def visualize_data(self, data, columns, title, xlabel, ylabel):
        data[columns].plot(figsize=(10, 5), title=title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def plot_correlation_matrix(self, data, title="Correlation Matrix"):
        correlation_matrix = data.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(title)
        plt.show()

    def backtest_lstm(self, predictions, actual_values):
        mse = mean_squared_error(actual_values, predictions)
        logging.info(f"LSTM Mean Squared Error: {mse}")

        # Profitability Calculation (Example: Cumulative Returns)
        actual_returns = actual_values[1:] - actual_values[:-1]
        predicted_returns = predictions[1:] - predictions[:-1]

        profitability = (predicted_returns * actual_returns).sum()
        logging.info(f"LSTM Profitability (Cumulative Returns): {profitability}")

        return mse, profitability

    def evaluate_models(self, stock_data):
        # LSTM Evaluation
        model, predictions = self.lstm_prediction(stock_data["close"])
        predictions = predictions.flatten()
        actual_values = stock_data["close"].values[-len(predictions):]
        lstm_mse, lstm_profit = self.backtest_lstm(predictions, actual_values)

        # Bayesian Networks Evaluation
        query_result = self.build_bayesian_network(stock_data)
        logging.info(f"Bayesian Network Query Result: {query_result}")

        # Reinforcement Learning Evaluation
        rewards = self.reinforcement_learning(env_name="CartPole-v1", episodes=50)
        avg_reward = sum(rewards) / len(rewards)
        logging.info(f"Reinforcement Learning Average Reward: {avg_reward}")

        # Comparison of Models
        comparison_results = {
            "LSTM": {"MSE": lstm_mse, "Profitability": lstm_profit},
            "Bayesian Network": {"Example Query": query_result},
            "Reinforcement Learning": {"Average Reward": avg_reward},
        }

        return comparison_results

    def visualize_comparison(self, comparison_results):
        lstm_metrics = comparison_results["LSTM"]
        mse_values = [lstm_metrics.get("MSE", 0)]
        profitability_values = [lstm_metrics.get("Profitability", 0)]
        rl_avg_reward = comparison_results["Reinforcement Learning"].get("Average Reward", 0)

        models_mse_profit = ["LSTM"]
        models_rl = ["Reinforcement Learning"]

        if any(mse_values):
            plt.figure(figsize=(10, 5))
            plt.bar(models_mse_profit, mse_values, label="MSE", color="blue", alpha=0.6)
            plt.title("Model MSE Comparison")
            plt.ylabel("MSE")
            plt.show()

        if any(profitability_values):
            plt.figure(figsize=(10, 5))
            plt.bar(models_mse_profit, profitability_values, label="Profitability", color="green", alpha=0.6)
            plt.title("Model Profitability Comparison")
            plt.ylabel("Profitability")
            plt.show()

        plt.figure(figsize=(10, 5))
        plt.bar(models_rl, [rl_avg_reward], label="Avg Reward", color="orange", alpha=0.6)
        plt.title("Reinforcement Learning Average Reward")
        plt.ylabel("Average Reward")
        plt.show()


if __name__ == "__main__":
    config = {
        "reddit_client_id": "MmsK7PahE1pwyzbLF2wBKQ",
        "reddit_client_secret": "zAFr9cMvdk1TacfbKiuLL2HH5I0X6A",
        "reddit_user_agent": "mmmirvari",
    }
    pipeline = FinancialDataPipeline(config)
    pipeline.fetch_reddit_headlines("stockmarket", 100, "data/sentiment/raw")
    pipeline.analyze_sentiment("data/sentiment/raw/stockmarket_headlines.csv", "data/sentiment/processed/stockmarket_sentiment.csv")

    pipeline.fetch_stock_data("AAPL", "2023-01-01", "2024-01-01", "data/stocks/AAPL.csv")
    stock_data = pd.read_csv("data/stocks/AAPL.csv", index_col="Date", parse_dates=True)
    stock_data = pipeline.add_technical_indicators(stock_data)

    pipeline.fetch_crypto_data("BTCUSDT", "1d", "2023-01-01", "2024-01-01", "data/crypto/BTCUSDT.csv")
    crypto_data = pd.read_csv("data/crypto/BTCUSDT.csv", index_col="Date", parse_dates=True)
    crypto_data = pipeline.add_technical_indicators(crypto_data)

    pipeline.plot_correlation_matrix(stock_data, title="Correlation Matrix of AAPL Features")
    pipeline.visualize_data(stock_data, ["close", "sma_20", "ema_20"], "AAPL Stock with Indicators", "Date", "Price")

    pipeline.plot_correlation_matrix(crypto_data, title="Correlation Matrix of BTCUSDT Features")
    pipeline.visualize_data(crypto_data, ["close", "sma_20", "ema_20"], "BTCUSDT with Indicators", "Date", "Price")

    stock_comparison_results = pipeline.evaluate_models(stock_data)
    print("Stock Comparison Results:", stock_comparison_results)

    crypto_comparison_results = pipeline.evaluate_models(crypto_data)
    print("Crypto Comparison Results:", crypto_comparison_results)

    pipeline.visualize_comparison(stock_comparison_results)
    pipeline.visualize_comparison(crypto_comparison_results)
