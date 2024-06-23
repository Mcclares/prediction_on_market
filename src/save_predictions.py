import pandas as pd
from src.predict import predict

if __name__ == "__main__":
    processed_data_path = "../data/processed/processed_data.csv"
    model_path = "../saved_models/model.keras"

    data = pd.read_csv(processed_data_path, parse_dates=['Date'], index_col='Date')
    data = data[['Close', 'SMA_50', 'SMA_200', 'RSI']]

    predictions = predict(data, model_path)

    # Сохранение предсказаний в CSV
    prediction_df = pd.DataFrame(predictions, index=data.index[-len(predictions):], columns=['Predicted_Close'])
    prediction_df.to_csv("../data/predictions/predictions.csv")

    print(predictions)