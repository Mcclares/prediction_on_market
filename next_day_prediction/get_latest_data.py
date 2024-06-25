import os
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from src.feature_engineering import create_features, generate_signals


def get_latest_data(tickers, end_date, years=2):
    """
    Получить данные за последние несколько лет для указанных тикеров.

    Parameters:
    tickers (list): Список тикеров.
    end_date (str): Конечная дата в формате 'YYYY-MM-DD'.
    years (int): Количество лет данных для загрузки.

    Returns:
    dict: Словарь с данными для каждого тикера.
    """
    end_date = pd.to_datetime(end_date)
    start_date = end_date - pd.DateOffset(years=years)
    all_data = {}

    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        data = data[['Close']]
        all_data[ticker] = data

    return all_data


def save_data(data, path, ticker):
    """
    Сохраняет данные в указанный путь с именем тикера.

    Parameters:
    data (pd.DataFrame): Данные для сохранения.
    path (str): Путь к папке для сохранения данных.
    ticker (str): Имя тикера.
    """
    file_path = os.path.join(path, f"{ticker}_latest_data.csv")
    data.to_csv(file_path)


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOG"]
    end_date = datetime.now().strftime('%Y-%m-%d')
    latest_data_path = "../data/latest/"
    processed_data_path = "../data/processed/"
    years = 12  # Количество лет данных для загрузки

    # Fetch and save latest data
    latest_data = get_latest_data(tickers, end_date, years)

    for ticker, data in latest_data.items():
        # Save raw data
        save_data(data, latest_data_path, ticker)

        # Preprocess and create features
        data = pd.read_csv(os.path.join(latest_data_path, f"{ticker}_latest_data.csv"), parse_dates=['Date'],
                           index_col='Date')
        data = create_features(data)
        data = generate_signals(data)

        # Save processed data
        processed_file_path = os.path.join(processed_data_path, f"{ticker}_processed_data.csv")
        data.to_csv(processed_file_path)
