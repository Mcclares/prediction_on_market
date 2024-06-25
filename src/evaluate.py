import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras._tf_keras.keras.models import load_model
from src.train import create_dataset


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
    test_size = len(X) - train_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    model = load_model(model_path)
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Обратное преобразование масштабированных данных
    train_predict_extended = np.zeros((train_predict.shape[0], scaled_data.shape[1]))
    test_predict_extended = np.zeros((test_predict.shape[0], scaled_data.shape[1]))

    # Вставка предсказанных значений в расширенные массивы
    train_predict_extended[:, 0] = train_predict[:, 0]
    test_predict_extended[:, 0] = test_predict[:, 0]

    # Добавление соответствующих входных данных (кроме предсказанной колонки)
    train_predict_extended[:, 1:] = X_train[:, -1, 1:]
    test_predict_extended[:, 1:] = X_test[:, -1, 1:]

    # Выполнение обратного масштабирования
    train_predict_inverse = scaler.inverse_transform(train_predict_extended)[:, 0]
    test_predict_inverse = scaler.inverse_transform(test_predict_extended)[:, 0]

    y_train_actual = scaler.inverse_transform(np.concatenate((y_train.reshape(-1, 1), X_train[:, -1, 1:]), axis=1))[:, 0]
    y_test_actual = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), X_test[:, -1, 1:]), axis=1))[:, 0]

    # Определение индексов для обучающих и тестовых предсказаний
    train_index = data.index[time_step:train_size + time_step]
    test_index = data.index[train_size + time_step:train_size + time_step + test_size]

    # Проверка длин массивов
    assert len(train_index) == len(train_predict_inverse), f"Length mismatch: train_index ({len(train_index)}), train_predict_inverse ({len(train_predict_inverse)})"
    assert len(test_index) == len(test_predict_inverse), f"Length mismatch: test_index ({len(test_index)}), test_predict_inverse ({len(test_predict_inverse)})"

    # Визуализация результатов
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close'], label='Исторические данные')
    plt.plot(train_index, train_predict_inverse, label='Прогноз (обучение)')
    plt.plot(test_index, test_predict_inverse, label='Прогноз (тест)')
    plt.legend()
    plt.show()