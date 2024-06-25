from keras._tf_keras.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


def create_dataset(dataset, time_step=1):
    X = []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:(i + time_step), :])
    return np.array(X)


def predict_next_day(data, model_path, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    if len(scaled_data) < time_step:
        raise ValueError(f"Not enough data to create a sequence of length {time_step}. Got {len(scaled_data)} samples.")

    last_sequence = scaled_data[-time_step:]
    last_sequence = np.expand_dims(last_sequence, axis=0)

    model = load_model(model_path)
    prediction = model.predict(last_sequence)

    prediction_extended = np.zeros((1, scaled_data.shape[1]))
    prediction_extended[0, 0] = prediction[0, 0]
    prediction_inverse = scaler.inverse_transform(prediction_extended)

    return prediction_inverse[0, 0]


if __name__ == "__main__":
    latest_data_path = "../data/latest/latest_data.csv"
    model_path = "../saved_models/regression_model.keras"

    data = pd.read_csv(latest_data_path, parse_dates=['Date'], index_col='Date')
    data = data[['Close', 'SMA_50', 'SMA_200', 'RSI']]

    next_day_prediction = predict_next_day(data, model_path)
    print(
        f"Предсказание на завтрашний день ({pd.to_datetime('2024-06-23') + pd.DateOffset(days=1)}): {next_day_prediction}")