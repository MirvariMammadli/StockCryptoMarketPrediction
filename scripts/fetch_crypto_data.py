import requests
import os
import pandas as pd

# Fetching cryptocurrency data using CoinGecko API
def fetch_crypto_data(crypto_id, days, save_path):
    print(f"Fetching data for {crypto_id} for the past {days} days...")
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart?vs_currency=usd&days={days}"
    response = requests.get(url)
    data = response.json()

    # Extracting prices
    prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    prices.to_csv(f"{save_path}/{crypto_id}.csv", index=False)
    print(f"Data saved for {crypto_id} at {save_path}/{crypto_id}.csv")

# Usage (example)
if __name__ == "__main__":
    cryptos = ['bitcoin', 'ethereum', 'dogecoin']  # List of crypto IDs
    for crypto in cryptos:
        fetch_crypto_data(crypto, 365, 'data/crypto')
