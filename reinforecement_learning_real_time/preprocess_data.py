import pandas as pd

def compute_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_features(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = compute_rsi(data['Close'], 14)
    data = data.dropna()
    return data

def preprocess_data(latest_data_path, processed_data_path, tickers):
    for ticker in tickers:
        data = pd.read_csv(f"{latest_data_path}{ticker}_latest_data.csv", parse_dates=['date'], index_col='date')
        data = create_features(data)
        data.to_csv(f"{processed_data_path}{ticker}_processed_data.csv")

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOG"]
    latest_data_path = "../data/latest/"
    processed_data_path = "../data/processed/"
    preprocess_data(latest_data_path, processed_data_path, tickers)