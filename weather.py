import requests
from datetime import datetime, timedelta
from dagster import op, job, schedule
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
        return max_temps
    else:
        print(f"Error fetching data: {response.status_code}")
        return None


@op
def train_model(data):
    max_temps = np.array(data).reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    max_temps_scaled = scaler.fit_transform(max_temps)

    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
        return np.array(sequences)

    seq_length = 10
    X_max = create_sequences(max_temps_scaled, seq_length)
    y_max = max_temps_scaled[seq_length:]

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model for max temperatures
    model.fit(X_max, y_max, epochs=20, batch_size=1, verbose=1)

    return model, scaler, X_max, y_max


@op
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
    data = fetch_data_from_api()
    model, scaler, X_max, y_max = train_model(data)
    mse, mae = evaluate_model(model, scaler, X_max, y_max)


@schedule(cron_schedule="0 0 * * *", job=weather_forecasting_pipeline)  # Run daily at midnight
def daily_weather_forecasting_schedule():
    return {}
    
if __name__ == "__main__":
    daily_weather_forecasting_schedule()
