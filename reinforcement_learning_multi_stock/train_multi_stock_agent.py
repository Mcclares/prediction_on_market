import pandas as pd
from stable_baselines3 import PPO
from multi_stock_trading_environment import MultiStockTradingEnvironment

if __name__ == "__main__":
    # Загрузка данных и подготовка окружения для нескольких акций
    tickers = ["AAPL", "MSFT", "GOOG"]
    data_frames = []
    for ticker in tickers:
        data = pd.read_csv(f"../data/processed/{ticker}_processed_data.csv", parse_dates=['Date'], index_col='Date')
        data_frames.append(data[['Close']])
    combined_data = pd.concat(data_frames, axis=1, keys=tickers)

    env = MultiStockTradingEnvironment(combined_data.values)

    # Обучение агента
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)

    # Сохранение модели
    model.save("ppo_multi_stock_trading_agent")