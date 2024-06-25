import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras._tf_keras.keras.models import load_model
from src.train import create_dataset

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def predict(data, model_path, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, _ = create_dataset(scaled_data, time_step)

    model = load_model(model_path)
    predictions = model.predict(X)

    # Добавление дополнительного измерения к предсказаниям
    predictions = predictions.reshape(-1, 1)

    # Создание расширенного массива для обратного масштабирования
    predictions_extended = np.zeros((predictions.shape[0], scaled_data.shape[1]))
    predictions_extended[:, 0] = predictions[:, 0]
    predictions_extended[:, 1:] = X[:, -1, 1:]

    # Выполнение обратного масштабирования
    predictions_inverse = scaler.inverse_transform(predictions_extended)[:, 0]
    return predictions_inverse


if __name__ == "__main__":
    processed_data_path = "../data/processed/processed_data.csv"
    model_path = "../saved_models/regression_model.keras"

    data = pd.read_csv(processed_data_path, parse_dates=['Date'], index_col='Date')
    data = data[['Close', 'SMA_50', 'SMA_200', 'RSI']]

    predictions = predict(data, model_path)

    print(predictions)
