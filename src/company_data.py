
import yfinance as yf
import os


def download_data(ticker, start_date, end_date, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = yf.download(ticker, start=start_date, end=end_date)
    data.to_csv(filepath)


if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2010-01-01"
    end_date = "2023-01-01"
    filepath = "../data/raw/raw_data.csv"
    download_data(ticker, start_date, end_date, filepath)
