import os
from alpha_vantage.timeseries import TimeSeries

import yfinance as yf


def fetch_realtime_data(tickers, start_date, end_date):
    all_data = {}
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date, interval='1m')
        data.reset_index(inplace=True)  # Ensure 'Datetime' is a column
        data.rename(columns={'Datetime': 'date'}, inplace=True)
        all_data[ticker] = data[['date', 'Close']]  # Ensure the 'date' column is included
    return all_data

def fetch_realtime_data(tickers, start_date, end_date):
    all_data = {}
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date, interval='1m')
        data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        data = data[['Close']]
        all_data[ticker] = data
    return all_data


def fetch_realtime_data_alpha(tickers, api_key, output_size='full'):
    ts = TimeSeries(key=api_key, output_format='pandas')
    all_data = {}
    for ticker in tickers:
        data, _ = ts.get_intraday(symbol=ticker, interval='1min', outputsize=output_size)
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = data[['Close']]
        all_data[ticker] = data
    return all_data


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOG"]
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')

    latest_data = fetch_realtime_data_alpha(tickers, api_key)

    latest_data_path = "../data/latest/"
    for ticker, data in latest_data.items():
        data.to_csv(f"{latest_data_path}{ticker}_latest_data.csv")
