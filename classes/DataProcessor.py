import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

class DataProcessor:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.scaler = MinMaxScaler()

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

    def add_technical_indicators(self, data):
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
