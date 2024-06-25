import os
import pandas as pd
from alpha_vantage.timeseries import TimeSeries


from dotenv import load_dotenv
load_dotenv()
# ALPHA_API_KEY = "7NADLRMVULDT3V6G"




def fetch_realtime_data(tickers, api_key, output_size='compact'):
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
    latest_data = fetch_realtime_data(tickers, api_key)

    latest_data_path = "../data/latest/"
    for ticker, data in latest_data.items():
        data.to_csv(f"{latest_data_path}{ticker}_latest_data.csv")


# def fetch_realtime_data(tickers, api_key):
#     base_url = "https://www.alphavantage.co/query"
#     latest_data = {}
#
#     for ticker in tickers:
#         params = {
#             "function": "TIME_SERIES_INTRADAY",
#             "symbol": ticker,
#             "interval": "1min",
#             "apikey": api_key
#         }
#
#         response = requests.get(base_url, params=params)
#         data = response.json()
#
#         # Assuming the returned data is in a dictionary under "Time Series (1min)"
#         time_series = data.get("Time Series (1min)", {})
#
#         # Convert the time series dictionary to a DataFrame
#         df = pd.DataFrame.from_dict(time_series, orient="index", dtype=float)
#         df.index = pd.to_datetime(df.index)
#
#         latest_data[ticker] = df
#
#     return latest_data