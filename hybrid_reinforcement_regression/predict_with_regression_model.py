# import numpy as np
# import pandas as pd
# from keras._tf_keras.keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler
#
# def predict_with_regression_model(data_path, model_path):
#     data = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
#     data = data[['Close']]
#
#     # Normalize the data
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(data)
#
#     # Prepare the input data for the LSTM model
#     time_step = 60
#     X = []
#     for i in range(time_step, len(scaled_data)):
#         X.append(scaled_data[i-time_step:i])
#     X = np.array(X)
#
#     # Load the trained model
#     model = load_model(model_path)
#
#     # Make predictions
#     predictions = model.predict(X)
#
#     # Inverse transform predictions
#     predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], 1))), axis=1))[:, 0]
#     return predictions
#
#
# if __name__ == "__main__":
#     predictions = predict_with_regression_model("../data/processed/processed_data.csv", "../saved_models/regression_model_realtime.keras")
#     pd.DataFrame(predictions, columns=['Predicted_Close']).to_csv("../data/predictions/regression_predictions.csv", index=False)
#

import numpy as np
import pandas as pd
from keras._tf_keras.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def predict_with_regression_model(data_path, model_path):
    data = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
    data = data[['Close']]

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Prepare the input data for the LSTM model
    time_step = 60
    X = []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i])
    X = np.array(X)

    # Print shapes for debugging
    print("Shape of X:", X.shape)

    # Load the trained model
    model = load_model(model_path)

    # Ensure input shape matches model's expected input shape
    input_shape = model.input_shape
    print("Model input shape:", input_shape)

    if X.shape[1:] != input_shape[1:]:
        raise ValueError(f"Input shape {X.shape[1:]} does not match model's expected input shape {input_shape[1:]}")

    # Make predictions
    predictions = model.predict(X)

    # Inverse transform predictions
    predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], 1))), axis=1))[:, 0]
    return predictions

if __name__ == "__main__":
    predictions = predict_with_regression_model("../data/processed/processed_data.csv", "../saved_models/regression_model_realtime.keras")
    pd.DataFrame(predictions, columns=['Predicted_Close']).to_csv("../data/predictions/regression_predictions.csv", index=False)