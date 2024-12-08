import yfinance as yf
import os

# Historical stock data (fetch and save)
def fetch_stock_data(ticker, start_date, end_date, save_path):
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data.to_csv(f"{save_path}/{ticker}.csv")
    print(f"Data saved for {ticker} at {save_path}/{ticker}.csv")

# Usage (example)
if __name__ == "__main__":
    stocks = ['AAPL', 'GOOGL', 'MSFT']  # List of tickers to fetch
    for stock in stocks:
        fetch_stock_data(stock, '2020-01-01', '2023-01-01', 'data/stocks')
