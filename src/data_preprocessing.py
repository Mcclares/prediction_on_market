
import pandas as pd

def preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data = data.dropna()
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

if __name__ == "__main__":
    raw_data_path = "../data/raw/raw_data.csv"
    processed_data_path = "../data/processed/processed_data.csv"
    data = preprocess_data(raw_data_path)
    data.to_csv(processed_data_path)
