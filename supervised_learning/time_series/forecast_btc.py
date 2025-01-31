import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping


def create_sequences(data, lookback=24, forecast_horizon=1):
    """Sequence Creation
    creates input-output pairs for the LSTM model.
    Each input sequence consists of the past 24 hours of data,
    and the output is the close price of the next hour.
    """
    X, y = [], []
    for i in range(lookback, len(data) - forecast_horizon):
        X.append(data[i - lookback:i])
        y.append(data[i + forecast_horizon - 1, 3])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    """create, train, and validate an LSTM model for BTC price forecasting.
    The model is trained using the Adam optimizer and mean-squared error (MSE) as the loss function.
    Early stopping is used to prevent overfitting.
    The trained model is saved as btc_forecast_lstm.h5
    """
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def main():
    """Run the scripts"""
    data = pd.read_csv('preprocessed_data.csv', index_col='Timestamp', parse_dates=True)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    lookback = 24
    forecast_horizon = 1
    X, y = create_sequences(data_scaled, lookback, forecast_horizon)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=3,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    model.save('btc_forecast_lstm.h5')


if __name__ == "__main__":
    main()
