import requests
from datetime import datetime, timedelta
from dagster import op, job, schedule, Out
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.keras
import pandas as pd

@op
def fetch_data_from_api():
    end_date = datetime.today()
    start_date = end_date - timedelta(days=80)

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    url = "https://api.open-meteo.com/v1/forecast"
    latitude, longitude = 35.1899, -0.6309  # Sidi Bel Abbes, Algeria

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max",   # Get Max temperature
        "temperature_unit": "celsius",  
        "start_date": start_date_str,  
        "end_date": end_date_str, 
        "timezone": "auto"  
    }

    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        max_temps = data['daily']['temperature_2m_max']  
        df = pd.DataFrame(max_temps)
        df.to_csv(f"data/weatherData-{end_date_str}.csv", index=False)
        
        return max_temps
    else:
        raise Exception(f"Error fetching data: {response.status_code}")


@op(
    out={"model": Out(), "scaler": Out(), "X_max": Out(), "y_max": Out()}
)
def train_model(data):
    seq_length=10
    epochs=20
    batch_size=1


    max_temps = np.array(data).reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    max_temps_scaled = scaler.fit_transform(max_temps)

    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
        return np.array(sequences)

    X_max = create_sequences(max_temps_scaled, seq_length)
    y_max = max_temps_scaled[seq_length:]

    # MLFlow Part
    with mlflow.start_run(run_name=f"weatherForecasting-LSTM-{datetime.today().strftime('%Y-%m-%d')}"):
        # Logging Dataset
        mlflow.log_artifact(f"data/weatherData-{datetime.today().strftime('%Y-%m-%d')}.csv")

        # Logging hyperparameters
        mlflow.log_param("seq_length", seq_length)
        mlflow.log_param("epochs", 20)
        mlflow.log_param("batch_size", 1)

        # Model 
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
        model.add(LSTM(50))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(X_max, y_max, epochs, batch_size, verbose=1)

        # Logging model
        mlflow.keras.log_model(model, "lstm_model")

        y_pred = model.predict(X_max)

        y_pred_rescaled = scaler.inverse_transform(y_pred)
        y_actual_rescaled = scaler.inverse_transform(y_max)

        mse = mean_squared_error(y_actual_rescaled, y_pred_rescaled)
        mae = mean_absolute_error(y_actual_rescaled, y_pred_rescaled)

        # Loggging metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)

        return model, scaler, X_max, y_max


@op(
    out={"mse": Out(), "mae": Out()}
)
def evaluate_model(model, scaler, X_max, y_max):
    # Predict using the model
    y_pred = model.predict(X_max)

    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_actual_rescaled = scaler.inverse_transform(y_max)

    # Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE)
    mse = mean_squared_error(y_actual_rescaled, y_pred_rescaled)
    mae = mean_absolute_error(y_actual_rescaled, y_pred_rescaled)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")

    return mse, mae


@job
def weather_forecasting_pipeline():
    max_temps = fetch_data_from_api()

    model, scaler, X_max, y_max = train_model(max_temps)

    mse, mae = evaluate_model(model, scaler, X_max, y_max)

@schedule(cron_schedule="0 0 * * *", job=weather_forecasting_pipeline) 
def daily_weather_forecasting_schedule():
    return {}
