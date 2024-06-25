import pandas as pd
import time
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from multi_stock_trading_environment import MultiStockTradingEnvironment
from hybrid_reinforcement_regression.preprocess_data import preprocess_data
from predict_with_regression_model import predict_with_regression_model
import numpy as np
from datetime import datetime, timedelta

from realtime_data_fetcher import fetch_realtime_data

def update_and_train_agent():
    tickers = ["AAPL", "MSFT", "GOOG"]
    latest_data_path = "../data/latest/"
    processed_data_path = "../data/processed/"

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    # Fetch new real-time data using yfinance
    latest_data = fetch_realtime_data(tickers, start_date, end_date)

    # Save latest data
    for ticker, data in latest_data.items():
        print(f"Latest data for {ticker}:")
        print(data.head())
        data.to_csv(f"{latest_data_path}{ticker}_latest_data.csv", index=False)

    # Preprocess the data
    preprocess_data(latest_data_path, processed_data_path, tickers)

    # Generate predictions using regression model
    combined_predictions = []
    for ticker in tickers:
        predictions = predict_with_regression_model(f"{processed_data_path}{ticker}_processed_data.csv",
                                                    "../saved_models/regression_model.keras")
        combined_predictions.append(predictions)

    min_length = min(len(pred) for pred in combined_predictions)

    combined_predictions = [pred[:min_length] for pred in combined_predictions]
    combined_predictions = np.column_stack(combined_predictions)

    data_frames = []
    for ticker in tickers:
        data = pd.read_csv(f"{processed_data_path}{ticker}_processed_data.csv", parse_dates=['date'], index_col='date')
        print(f"Processed data for {ticker}:")
        print(data.head())
        data_frames.append(data[['Close']])
    combined_data = pd.concat(data_frames, axis=1, keys=tickers)

    combined_data = combined_data.iloc[-min_length:]

    if np.any(np.isnan(combined_data.values)) or np.any(np.isnan(combined_predictions)):
        raise ValueError("Combined data or predictions contain NaN values. Please check your preprocessing steps.")

    env = DummyVecEnv([lambda: MultiStockTradingEnvironment(combined_data.values, combined_predictions)])

    try:
        model = PPO.load("ppo_multi_stock_trading_agent")
        model.set_env(env)
    except FileNotFoundError:
        model = PPO('MlpPolicy', env, verbose=1, learning_rate=1e-5)

    model.learn(total_timesteps=1000, reset_num_timesteps=False)

    model.save("ppo_multi_stock_trading_agent")

if __name__ == "__main__":
    while True:
        update_and_train_agent()
        time.sleep(60 * 60)