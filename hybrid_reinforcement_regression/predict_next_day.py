import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from multi_stock_trading_environment import MultiStockTradingEnvironment
from predict_with_regression_model import predict_with_regression_model
from datetime import datetime, timedelta
from visualize_agent_results import load_latest_data


def predict_next_day(env, model, data):
    obs = env.reset()
    for _ in range(len(data)):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break

    # Prediction for the next day
    next_day_action, _ = model.predict(obs)
    return next_day_action


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOG"]
    latest_data_path = "../data/latest/"

    # Load the latest data
    combined_data = load_latest_data(tickers, latest_data_path)

    # Generate predictions using the regression model
    combined_predictions = []
    for ticker in tickers:
        predictions = predict_with_regression_model(f"{latest_data_path}{ticker}_latest_data.csv",
                                                    "../saved_models/regression_model_realtime.keras")
        combined_predictions.append(predictions)
    combined_predictions = np.column_stack(combined_predictions)

    # Prepare the environment
    env = MultiStockTradingEnvironment(combined_data.values, combined_predictions)

    # Load the trained model
    model = PPO.load("ppo_multi_stock_trading_agent")

    # Predict actions for the next day
    next_day_action = predict_next_day(env, model, combined_data.values)

    tomorrow = datetime.now() + timedelta(days=1)
    tomorrow_str = pd.to_datetime(tomorrow).strftime('%Y-%m-%d')
    print(f"Predicted actions for the next day ({tomorrow_str}):")
    for ticker, action in zip(tickers, next_day_action):
        action_str = "hold" if action == 0 else "buy" if action == 1 else "sell"
        print(f"{ticker}: {action_str}")