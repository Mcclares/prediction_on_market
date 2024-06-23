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
    Генерирует сигналы покупки и продажи на основе SMA и RSI.

    Параметры:
    data (pd.DataFrame): Данные с признаками SMA и RSI.

    Возвращает:
    pd.DataFrame: Данные с добавленными сигналами.
    """
    data['Buy_Signal'] = ((data['SMA_50'] > data['SMA_200']) & (data['SMA_50'].shift(1) <= data['SMA_200'].shift(1))) | (data['RSI'] < 30)
    data['Sell_Signal'] = ((data['SMA_50'] < data['SMA_200']) & (data['SMA_50'].shift(1) >= data['SMA_200'].shift(1))) | (data['RSI'] > 70)
    return data
    data = generate_signals(data)
    data.to_csv(processed_data_path)


if __name__ == "__main__":
    processed_data_path = "../data/processed/processed_data.csv"
    data = pd.read_csv(processed_data_path, parse_dates=['Date'], index_col='Date')
    data = create_features(data)
    data = generate_signals(data)
    data.to_csv(processed_data_path)




#Более улучшенная модель
import pandas as pd
import numpy as np


# def compute_ema(data, window):
#     return data.ewm(span=window, adjust=False).mean()
#
#
# def compute_bollinger_bands(data, window):
#     sma = data.rolling(window).mean()
#     std = data.rolling(window).std()
#     upper_band = sma + 2 * std
#     lower_band = sma - 2 * std
#     return sma, upper_band, lower_band
#
#
# def compute_macd(data, short_window=12, long_window=26, signal_window=9):
#     short_ema = compute_ema(data, short_window)
#     long_ema = compute_ema(data, long_window)
#     macd = short_ema - long_ema
#     signal = compute_ema(macd, signal_window)
#     return macd, signal
#
#
# def compute_stochastic_oscillator(data, window=14):
#     low_min = data['Low'].rolling(window=window).min()
#     high_max = data['High'].rolling(window=window).max()
#     k = 100 * (data['Close'] - low_min) / (high_max - low_min)
#     d = k.rolling(window=3).mean()
#     return k, d
#
#
# def generate_signals(data):
#     data['SMA_50'] = data['Close'].rolling(window=50).mean()
#     data['SMA_200'] = data['Close'].rolling(window=200).mean()
#     data['EMA_50'] = compute_ema(data['Close'], 50)
#
#     data['RSI'] = compute_rsi(data['Close'], 14)
#
#     data['SMA'], data['Upper_Band'], data['Lower_Band'] = compute_bollinger_bands(data['Close'], 20)
#
#     data['MACD'], data['Signal_Line'] = compute_macd(data['Close'])
#
#     data['Stochastic_K'], data['Stochastic_D'] = compute_stochastic_oscillator(data)
#
#     data['Buy_Signal'] = (
#             ((data['SMA_50'] > data['SMA_200']) & (data['SMA_50'].shift(1) <= data['SMA_200'].shift(1))) |
#             (data['RSI'] < 30) |
#             ((data['MACD'] > data['Signal_Line']) & (data['MACD'].shift(1) <= data['Signal_Line'].shift(1))) |
#             ((data['Stochastic_K'] < 20) & (data['Stochastic_D'] < 20))
#     )
#
#     data['Sell_Signal'] = (
#             ((data['SMA_50'] < data['SMA_200']) & (data['SMA_50'].shift(1) >= data['SMA_200'].shift(1))) |
#             (data['RSI'] > 70) |
#             ((data['MACD'] < data['Signal_Line']) & (data['MACD'].shift(1) >= data['Signal_Line'].shift(1))) |
#             ((data['Stochastic_K'] > 80) & (data['Stochastic_D'] > 80))
#     )
#
#     return data
#
#
# def create_features(data):
#     data['SMA_50'] = data['Close'].rolling(window=50).mean()
#     data['SMA_200'] = data['Close'].rolling(window=200).mean()
#     data['RSI'] = compute_rsi(data['Close'], 14)
#     data = data.dropna()
#     return data
#
#
# def compute_rsi(data, window):
#     delta = data.diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
#     rs = gain / loss
#     return 100 - (100 / (1 + rs))
