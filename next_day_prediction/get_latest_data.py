import yfinance as yf
import pandas as pd
from src.feature_engineering import create_features, generate_signals


def get_latest_data(ticker, end_date, days=730):
    start_date = pd.to_datetime(end_date) - pd.DateOffset(days=days)
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


if __name__ == "__main__":
    ticker = "AAPL"
    end_date = "2024-06-23"
    latest_data_path = "../data/latest/latest_data.csv"

    data = get_latest_data(ticker, end_date, days=730)
    data.to_csv(latest_data_path)

    data = pd.read_csv(latest_data_path, parse_dates=['Date'], index_col='Date')
    data = create_features(data)
    data = generate_signals(data)
    data.to_csv(latest_data_path)
