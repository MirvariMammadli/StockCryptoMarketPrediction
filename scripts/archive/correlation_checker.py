import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

stock_data = pd.read_csv("data/stocks/AAPL.csv")
crypto_data = pd.read_csv("data/crypto/bitcoin.csv")

merged_data = pd.concat(
    [stock_data["Close"], crypto_data["price"]],
    axis=1,
    keys=["AAPL_Close", "BTC_Close"],
)


correlation_matrix = merged_data.corr()

# correlation between Apple stock price and bitcoin
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix Between Stock and Crypto Prices")
plt.show()



print("Missing values in stock data:")
print(stock_data.isnull().sum())

print("\nMissing values in crypto data:")
print(crypto_data.isnull().sum())

# Filling the missing values with forward-fill
stock_data.fillna(method="ffill", inplace=True)
crypto_data.fillna(method="ffill", inplace=True)


print("\nMissing values after handling:")
print(stock_data.isnull().sum())
print(crypto_data.isnull().sum())

# Boxplot for stock prices
plt.figure(figsize=(10, 5))
sns.boxplot(stock_data["Close"])
plt.title("Boxplot for AAPL Closing Prices")
plt.show()

# Boxplot for cryptocurrency prices
plt.figure(figsize=(10, 5))
sns.boxplot(crypto_data["price"], color="orange")
plt.title("Boxplot for BTC Closing Prices")
plt.show()
