import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("vader_lexicon")

# Load Reddit sentiment data
reddit_data = pd.read_csv("data/sentiment/raw/stockmarket_headlines.csv")

sia = SentimentIntensityAnalyzer()

def preprocess_headlines(headline):
    tokens = word_tokenize(headline)
    tokens = [word.lower() for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

reddit_data["processed_title"] = reddit_data["title"].apply(preprocess_headlines)

# Compute sentiment scores
reddit_data["sentiment_score"] = reddit_data["processed_title"].apply(
    lambda x: sia.polarity_scores(x)["compound"]
)


reddit_data.to_csv("data/sentiment/processed/stockmarket_sentiment.csv", index=False)
print("Processed sentiment data saved!")
