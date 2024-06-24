from stable_baselines3 import PPO
from trading_environment import TradingEnvironment
import pandas as pd

if __name__ == "__main__":
    processed_data_path = "../data/processed/processed_data.csv"
    data = pd.read_csv(processed_data_path, parse_dates=['Date'], index_col='Date')
    data = data[['Close', 'SMA_50', 'SMA_200', 'RSI']].values

    env = TradingEnvironment(data)

    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)

    model.save("ppo_trading_agent")