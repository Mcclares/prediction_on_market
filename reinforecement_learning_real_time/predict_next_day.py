import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from reinforcement_learning_multi_stock.multi_stock_trading_environment import MultiStockTradingEnvironment
from dotenv import load_dotenv
load_dotenv()

def load_latest_data(tickers, latest_data_path):
    data_frames = []
    for ticker in tickers:
        data = pd.read_csv(f"{latest_data_path}{ticker}_latest_data.csv", parse_dates=['Date'], index_col='Date')
        data_frames.append(data[['Close']])
    combined_data = pd.concat(data_frames, axis=1, keys=tickers)
    return combined_data


def predict_next_day(env, model, data):
    obs = env.reset()
    for _ in range(len(data)):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break

    # Прогноз на следующий день
    next_day_action, _ = model.predict(obs)
    return next_day_action


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOG"]
    latest_data_path = "../data/latest/"

    # Загрузка последних данных
    combined_data = load_latest_data(tickers, latest_data_path)

    # Подготовка окружения
    env = MultiStockTradingEnvironment(combined_data.values)

    # Загрузка обученной модели
    model = PPO.load("ppo_multi_stock_trading_agent")

    # Прогнозирование действий на следующий день
    next_day_action = predict_next_day(env, model, combined_data.values)

    print(f"Прогноз действий на следующий день ({pd.to_datetime('2024-06-24')}):")
    for ticker, action in zip(tickers, next_day_action):
        action_str = "держать" if action == 0:
        иначе
        "купить"
        если
        action == 1
        иначе
        "продать"
        print(f"{ticker}: {action_str}")