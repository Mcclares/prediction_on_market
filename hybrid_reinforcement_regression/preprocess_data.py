import os
import pandas as pd


def compute_rsi(data, window):
    """
    Compute the Relative Strength Index (RSI) for a given data series.

    Parameters:
    data (pd.Series): Time series data of prices.
    window (int): The window size to compute RSI.

    Returns:
    pd.Series: The RSI values.
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def create_features(data):
    """
    Add technical indicators (SMA and RSI) to the data.

    Parameters:
    data (pd.DataFrame): The input data with 'Close' prices.

    Returns:
    pd.DataFrame: The data with additional features.
    """
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = compute_rsi(data['Close'], 14)
    data = data.dropna()
    return data


def generate_signals(data):
    """
    Generate buy and sell signals based on SMA and RSI.

    Parameters:
    data (pd.DataFrame): Data with SMA and RSI features.

    Returns:
    pd.DataFrame: Data with added signals.
    """
    data['Buy_Signal'] = ((data['SMA_50'] > data['SMA_200']) & (
                data['SMA_50'].shift(1) <= data['SMA_200'].shift(1))) | (data['RSI'] < 30)
    data['Sell_Signal'] = ((data['SMA_50'] < data['SMA_200']) & (
                data['SMA_50'].shift(1) >= data['SMA_200'].shift(1))) | (data['RSI'] > 70)
    return data


# def preprocess_data(latest_data_path, processed_data_path, tickers):
#     """
#     Preprocess data for each ticker and save processed data with additional features and signals.
#
#     Parameters:
#     latest_data_path (str): Path to the folder with latest data.
#     processed_data_path (str): Path to the folder to save processed data.
#     tickers (list): List of tickers to process.
#     """
#     for ticker in tickers:
#         # Load latest data
#         latest_data_file = os.path.join(latest_data_path, f"{ticker}_latest_data.csv")
#         df = pd.read_csv(latest_data_file, parse_dates=['date'], index_col='date')
#
#         # Fill missing values
#         df = df.ffill().bfill()
#
#         # Create features
#         df = create_features(df)
#
#         # Generate signals
#         df = generate_signals(df)
#
#         # Save processed data
#         processed_data_file = os.path.join(processed_data_path, f"{ticker}_processed_data.csv")
#         df.to_csv(processed_data_file)

def preprocess_data(latest_data_path, processed_data_path, tickers):
    """
    Preprocess data for each ticker and save processed data with additional features and signals.

    Parameters:
    latest_data_path (str): Path to the folder with latest data.
    processed_data_path (str): Path to the folder to save processed data.
    tickers (list): List of tickers to process.
    """
    for ticker in tickers:
        # Load latest data
        latest_data_file = os.path.join(latest_data_path, f"{ticker}_latest_data.csv")
        df = pd.read_csv(latest_data_file, parse_dates=['Datetime'], index_col='Datetime')
        df.index.name = 'date'  # Ensure the index name is 'date'

        print(f"Loaded data for {ticker}:")
        print(df.head())

        # Fill missing values
        df = df.ffill().bfill()

        # Create features
        df = create_features(df)

        # Generate signals
        df = generate_signals(df)

        # Save processed data
        processed_data_file = os.path.join(processed_data_path, f"{ticker}_processed_data.csv")
        df.to_csv(processed_data_file)
        print(f"Processed data for {ticker} saved:")
        print(df.head())


# Пример использования функции:
# preprocess_data("../data/latest/", "../data/processed/", ["AAPL", "MSFT", "GOOG"])

if __name__ == "__main__":
    latest_data_path = "../data/latest/"
    processed_data_path = "../data/processed/"
    tickers = ["AAPL", "MSFT", "GOOG"]
    preprocess_data(latest_data_path, processed_data_path, tickers)
