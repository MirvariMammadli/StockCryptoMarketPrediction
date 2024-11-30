import pandas as pd
import matplotlib.pyplot as plt

stock_data = pd.read_csv("data/stocks/AAPL.csv", index_col="Date", parse_dates=True)
crypto_data = pd.read_csv("data/crypto/bitcoin.csv", index_col="timestamp", parse_dates=True)

# Line plot for stock prices
plt.figure(figsize=(10, 5))
plt.plot(stock_data["Close"], label="AAPL Closing Price")
plt.title("AAPL Stock Price Trend")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

# Line plot for cryptocurrency prices
plt.figure(figsize=(10, 5))
plt.plot(crypto_data["price"], label="BTC Closing Price", color="orange")
plt.title("BTC Cryptocurrency Price Trend")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

#Histogram STock Prices
plt.hist(stock_data["Close"], bins=50, alpha=0.7, label="AAPL Closing Price")
plt.title("Distribution of AAPL Stock Prices")
plt.xlabel("Price (USD)")
plt.ylabel("Frequency")
plt.legend()
plt.show()


#HGistogram Crypto Prices
plt.figure(figsize=(10, 5))
plt.hist(crypto_data["price"], bins=50, alpha=0.7, color="orange", label="BTC Closing Price")
plt.title("Distribution of BTC Cryptocurrency Prices")
plt.xlabel("Price (USD)")
plt.ylabel("Frequency")
plt.legend()
plt.show()
