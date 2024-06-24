import pandas as pd
from stable_baselines3 import PPO
from trading_environment import TradingEnvironment

def trade(env, model, data):
    obs = env.reset()
    for _ in range(len(data)):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break


if __name__ == "__main__":
    processed_data_path = "../data/processed/processed_data.csv"
    data = pd.read_csv(processed_data_path, parse_dates=['Date'], index_col='Date')
    data = data[['Close', 'SMA_50', 'SMA_200', 'RSI']].values

    env = TradingEnvironment(data)

    model = PPO.load("ppo_trading_agent")

    # Торговля с использованием агента
    trade(env, model, data)