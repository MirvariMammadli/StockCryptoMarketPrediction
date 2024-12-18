import requests
import pandas as pd

def get_binance_historical_data(symbol, interval, start_date, end_date):
    base_url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(pd.Timestamp(start_date).timestamp() * 1000),
        "endTime": int(pd.Timestamp(end_date).timestamp() * 1000),
        "limit": 1000
    }
    
    response = requests.get(base_url, params=params)
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
    df = df.astype(float)

    return df

if __name__ == "__main__":
    symbol = "BTCUSDT"
    interval = "1d"
    start_date = "2023-01-01"
    end_date = "2024-10-01"
    data = get_binance_historical_data(symbol, interval, start_date, end_date)
    data.to_csv(f"data/crypto/{symbol}.csv")
    print(f"Data saved for {symbol}!")
