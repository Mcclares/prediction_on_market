
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.model import create_model
import numpy as np


def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), :]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)


if __name__ == "__main__":
    processed_data_path = "../data/processed/processed_data.csv"
    model_path = "../saved_models/regression_model.keras"

    data = pd.read_csv(processed_data_path, parse_dates=['Date'], index_col='Date')
    data = data[['Close', 'SMA_50', 'SMA_200', 'RSI']]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    time_step = 60
    X, y = create_dataset(scaled_data, time_step)

    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    model = create_model((time_step, X.shape[2]))
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

    model.save(model_path)




# # scripts/train_model.py
#
# import pandas as pd
# from src.data_preprocessing import preprocess_data
# from src.feature_engineering import create_features
# from src.train import train
#
# if __name__ == "__main__":
#     raw_data_path = "data/raw/raw_data.csv"
#     processed_data_path = "data/processed/processed_data.csv"
#
#     # Шаг 1: Загрузка данных
#     data = pd.read_csv(raw_data_path, parse_dates=['Date'], index_col='Date')
#
#     # Шаг 2: Предварительная обработка данных
#     data = preprocess_data(data)
#
#     # Шаг 3: Создание признаков
#     data = create_features(data)
#
#     # Шаг 4: Сохранение обработанных данных
#     data.to_csv(processed_data_path)
#
#     # Шаг 5: Обучение модели
#     model = train_model(data)
#
#     # Шаг 6: Сохранение модели
#     model.save("models/trained_model.h5")