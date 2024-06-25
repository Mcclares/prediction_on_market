# import pandas as pd
# import time
# import os
# from stable_baselines3 import PPO
# from reinforcement_learning_multi_stock.multi_stock_trading_environment import MultiStockTradingEnvironment
# from realtime_data_fetcher import fetch_realtime_data
# from dotenv import load_dotenv
# load_dotenv()
#
# def update_and_train_agent():
#     # Обновление данных
#     tickers = ["AAPL", "MSFT", "GOOG"]
#     api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
#     data_path = "../data/processed/"
#
#     # Fetch new real-time data
#     latest_data = fetch_realtime_data(tickers, api_key)
#
#     # Append new data to existing data and save
#     for ticker, new_data in latest_data.items():
#         data_file = f"{data_path}{ticker}_processed_data.csv"
#         if os.path.exists(data_file):
#             existing_data = pd.read_csv(data_file, parse_dates=['Date'], index_col='Date')
#             combined_data = pd.concat([existing_data, new_data])
#             combined_data = combined_data[~combined_data.index.duplicated(keep='last')]  # Remove duplicates
#         else:
#             combined_data = new_data
#         combined_data.to_csv(data_file)
#
#     # Load updated data and prepare environment
#     data_frames = []
#     for ticker in tickers:
#         data = pd.read_csv(f"{data_path}{ticker}_processed_data.csv", parse_dates=['Date'], index_col='Date')
#         data_frames.append(data[['Close']])
#     combined_data = pd.concat(data_frames, axis=1, keys=tickers)
#
#     env = MultiStockTradingEnvironment(combined_data.values)
#
#     # Load existing model or create new
#     try:
#         model = PPO.load("ppo_multi_stock_trading_agent")
#     except FileNotFoundError:
#         model = PPO('MlpPolicy', env, verbose=1)
#
#     # Continue training the agent
#     model.set_env(env)
#     model.learn(total_timesteps=1000, reset_num_timesteps=False)
#
#     # Save the model
#     model.save("ppo_multi_stock_trading_agent")
#
#
# if __name__ == "__main__":
#     while True:
#         update_and_train_agent()
#         time.sleep(60 * 60)  # Wait for 1 hour before fetching new data and retraining


import pandas as pd
import time
import os
from stable_baselines3 import PPO
from multi_stock_trading_environment import MultiStockTradingEnvironment
from realtime_data_fetcher import fetch_realtime_data
from preprocess_data import preprocess_data


def update_and_train_agent():
    # Обновление данных
    tickers = ["AAPL", "MSFT", "GOOG"]
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    latest_data_path = "../data/latest/"
    processed_data_path = "../data/processed/"

    # Fetch new real-time data
    latest_data = fetch_realtime_data(tickers, api_key)

    # Save latest data
    for ticker, data in latest_data.items():
        data.to_csv(f"{latest_data_path}{ticker}_latest_data.csv")

    # Preprocess the data
    preprocess_data(latest_data_path, processed_data_path, tickers)

    # Load updated data and prepare environment
    data_frames = []
    for ticker in tickers:
        data = pd.read_csv(f"{processed_data_path}{ticker}_processed_data.csv", parse_dates=['date'], index_col='date')
        data_frames.append(data[['Close', 'SMA_50', 'SMA_200', 'RSI']])
    combined_data = pd.concat(data_frames, axis=1, keys=tickers)

    env = MultiStockTradingEnvironment(combined_data.values)

    # Load existing model or create new
    try:
        model = PPO.load("ppo_multi_stock_trading_agent")
    except FileNotFoundError:
        model = PPO('MlpPolicy', env, verbose=1)

    # Continue training the agent
    model.set_env(env)
    model.learn(total_timesteps=1000, reset_num_timesteps=False)

    # Save the model
    model.save("ppo_multi_stock_trading_agent")


if __name__ == "__main__":
    while True:
        update_and_train_agent()
        time.sleep(60 * 60)  # Wait for 1 hour before fetching new data and retraining