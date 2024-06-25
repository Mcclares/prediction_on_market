import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from multi_stock_trading_environment import MultiStockTradingEnvironment
from predict_with_regression_model import predict_with_regression_model


def load_latest_data(tickers, latest_data_path):
    data_frames = []
    for ticker in tickers:
        data = pd.read_csv(f"{latest_data_path}{ticker}_latest_data.csv", parse_dates=['Date'], index_col='Date')
        data_frames.append(data[['Close']])
    combined_data = pd.concat(data_frames, axis=1, keys=tickers)
    return combined_data


def visualize_agent(env, model, data, tickers):
    obs = env.reset()
    balances = []
    profits = []
    holdings = []

    for _ in range(len(data)):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        balances.append(env.balance)
        profits.append(env.balance + np.sum(env.shares_held * env.data[env.current_step, :]) - env.initial_balance)
        holdings.append(env.shares_held.copy())
        if done:
            break

    holdings = np.array(holdings)
    dates = pd.date_range(start=data.index[0], periods=len(balances), freq='D')

    fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    for i, ticker in enumerate(tickers):
        axs[0].plot(dates, data[ticker]['Close'], label=f'{ticker} Price')
    axs[0].set_ylabel('Price')
    axs[0].legend()

    axs[1].plot(dates, balances, label='Balance')
    axs[1].set_ylabel('Balance')
    axs[1].legend()

    axs[2].plot(dates, profits, label='Profit')
    axs[2].set_ylabel('Profit')
    axs[2].legend()

    for i, ticker in enumerate(tickers):
        axs[3].plot(dates, holdings[:, i], label=f'{ticker} Holdings')
    axs[3].set_ylabel('Holdings')
    axs[3].legend()

    plt.xlabel('Date')
    plt.show()


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOG"]
    latest_data_path = "../data/latest/"

    # Load the latest data
    combined_data = load_latest_data(tickers, latest_data_path)

    # Generate predictions using the regression model
    combined_predictions = []
    for ticker in tickers:
        predictions = predict_with_regression_model(f"{latest_data_path}{ticker}_latest_data.csv",
                                                    "regression_model.h5")
        combined_predictions.append(predictions)
    combined_predictions = np.column_stack(combined_predictions)

    # Prepare the environment
    env = MultiStockTradingEnvironment(combined_data.values, combined_predictions)

    # Load the trained model
    model = PPO.load("ppo_multi_stock_trading_agent")

    # Visualize agent's performance
    visualize_agent(env, model, combined_data, tickers)