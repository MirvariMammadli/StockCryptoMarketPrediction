import pandas as pd
from classes.DataFetcher import DataFetcher
from classes.DataProcessor import DataProcessor
from classes.ModelEvaluator import ModelEvaluator

class FinancialDataPipeline:
    def __init__(self, config):
        self.config = config
        self.fetcher = DataFetcher(config)
        self.processor = DataProcessor()
        self.evaluator = ModelEvaluator()

    def run(self):
        # Fetch Reddit data
        self.fetcher.fetch_reddit_headlines("stockmarket", 100, "data/sentiment/raw")
        self.processor.analyze_sentiment("data/sentiment/raw/stockmarket_headlines.csv", "data/sentiment/processed/stockmarket_sentiment.csv")

        # Fetch and process stock data
        self.fetcher.fetch_stock_data("AAPL", "2023-01-01", "2024-01-01", "data/stocks/AAPL.csv")
        stock_data = pd.read_csv("data/stocks/AAPL.csv", index_col="Date", parse_dates=True)
        stock_data = self.processor.add_technical_indicators(stock_data)

        # Fetch and process cryptocurrency data
        self.fetcher.fetch_crypto_data("BTCUSDT", "1d", "2023-01-01", "2024-01-01", "data/crypto/BTCUSDT.csv")
        crypto_data = pd.read_csv("data/crypto/BTCUSDT.csv", index_col="Date", parse_dates=True)
        crypto_data = self.processor.add_technical_indicators(crypto_data)

        # Perform LSTM predictions and backtesting on stock data
        model, predictions = self.evaluator.lstm_prediction(stock_data["close"])
        actual_values = stock_data["close"].values[-len(predictions):]
        self.evaluator.backtest_lstm(predictions.flatten(), actual_values)

        # Perform reinforcement learning
        rewards = self.evaluator.reinforcement_learning(env_name="CartPole-v1", episodes=50)
        print(f"Average Reward over 50 episodes: {sum(rewards) / len(rewards):.2f}")

        model, predictions = self.evaluator.lstm_prediction(crypto_data["close"])
        actual_values = crypto_data["close"].values[-len(predictions):]
        self.evaluator.backtest_lstm(predictions.flatten(), actual_values)

        # Perform reinforcement learning
        rewards = self.evaluator.reinforcement_learning(env_name="CartPole-v1", episodes=50)
        print(f"Average Reward over 50 episodes: {sum(rewards) / len(rewards):.2f}")
        
        # Visualizations
        self.evaluator.plot_correlation_matrix(stock_data, title="Correlation Matrix for Stock Data")
        self.evaluator.visualize_data(stock_data, ["close", "sma_20", "ema_20"], "Stock Prices with Indicators", "Date", "Price")

        self.evaluator.plot_correlation_matrix(crypto_data, title="Correlation Matrix for Crypto Data")
        self.evaluator.visualize_data(crypto_data, ["close", "sma_20", "ema_20"], "Crypto Prices with Indicators", "Date", "Price")
        
        stock_comparison_results = self.evaluator.evaluate_models(stock_data)
        crypto_comparison_results = self.evaluator.evaluate_models(crypto_data)

        self.evaluator.visualize_comparison(stock_comparison_results, crypto_comparison_results)
