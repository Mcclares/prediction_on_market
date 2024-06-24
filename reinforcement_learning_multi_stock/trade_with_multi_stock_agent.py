import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from multi_stock_trading_environment import MultiStockTradingEnvironment


def trade(env, model, data, tickers):
    obs = env.reset()
    for _ in range(len(data)):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break

    # Анализ результатов
    total_profit = env.balance + np.sum(env.shares_held * env.data[env.current_step, :]) - env.initial_balance
    print(f"Total profit: {total_profit}")
    for i, ticker in enumerate(tickers):
        print(f"{ticker} profit: {(env.shares_held[i] * env.data[env.current_step, i]) - env.initial_balance}")


if __name__ == "__main__":
    # Загрузка данных и подготовка окружения для нескольких акций
    tickers = ["AAPL", "MSFT", "GOOG"]
    data_frames = []
    for ticker in tickers:
        data = pd.read_csv(f"../data/processed/{ticker}_processed_data.csv", parse_dates=['Date'], index_col='Date')
        data_frames.append(data[['Close']])
    combined_data = pd.concat(data_frames, axis=1, keys=tickers)

    env = MultiStockTradingEnvironment(combined_data.values)

    # Загрузка обученной модели
    model = PPO.load("ppo_multi_stock_trading_agent")

    # Торговля с использованием агента
    trade(env, model, combined_data.values, tickers)