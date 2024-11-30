from sklearn.preprocessing import MinMaxScaler
import pandas as pd

scaler = MinMaxScaler()
stock_data = pd.read_csv("data/stocks/AAPL.csv")

stock_data["Normalized_Close"] = scaler.fit_transform(stock_data["Close"].values.reshape(-1, 1))

stock_data.to_csv("data/stocks/AAPL_normalized.csv")
print("Data normalized and saved.")

stock_data["SMA_20"] = stock_data["Normalized_Close"].rolling(window=20).mean() 
stock_data["EMA_20"] = stock_data["Normalized_Close"].ewm(span=20, adjust=False).mean() 

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

stock_data["RSI_14"] = calculate_rsi(stock_data["Normalized_Close"])

short_ema = stock_data["Normalized_Close"].ewm(span=12, adjust=False).mean() 
long_ema = stock_data["Normalized_Close"].ewm(span=26, adjust=False).mean() 
stock_data["MACD"] = short_ema - long_ema
stock_data["Signal_Line"] = stock_data["MACD"].ewm(span=9, adjust=False).mean()  

stock_data.to_csv("data/stocks/AAPL_features.csv")
print("Feature engineering complete and data saved.")
