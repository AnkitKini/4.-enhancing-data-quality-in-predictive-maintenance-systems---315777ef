
# 1. ENVIRONMENT SETUP

!pip install pyod fancyimpute

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from fancyimpute import KNN
from scipy.signal import savgol_filter

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error


# 2. LOAD DATASETS


# NASA CMAPSS dataset
column_names = [
    'unit_number', 'time_in_cycles',
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
    'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20', 'sensor_21'
]

df_train = pd.read_csv("/content/sample_data/train_FD001.txt", sep="\s+", header=None)
df_train.dropna(axis=1, how='all', inplace=True)
df_train.columns = column_names


# Validation dataset
df_val = pd.read_csv("/content/sample_data/predictive_maintenance.csv")


# 3. PREPROCESSING


# Label RUL for NASA dataset
rul = df_train.groupby('unit_number')['time_in_cycles'].max().reset_index()
rul.columns = ['unit_number', 'max_cycle']
df_train = df_train.merge(rul, on='unit_number')
df_train['RUL'] = df_train['max_cycle'] - df_train['time_in_cycles']
df_train.drop('max_cycle', axis=1, inplace=True)

# Select informative features for NASA dataset
train_features = [
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7',
    'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12',
    'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'
]

# IMPORTANT: Validation must use same features as training for consistent input size
# If df_val doesn't have these columns, you need to process accordingly.
# For demonstration, we use only df_train here for both train and val split.

# Normalize with a single scaler (fit on train, transform on val)
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(df_train[train_features])


# 4. ANOMALY DETECTION

iso = IsolationForest(contamination=0.01)
anomalies = iso.fit_predict(train_scaled)
df_clean = df_train[anomalies == 1].reset_index(drop=True)
X_clean = scaler.transform(df_clean[train_features])
y_clean = df_clean['RUL']


# 5. FEATURE ENGINEERING (OPTIONAL)

df_clean['sensor2_mean'] = df_clean['sensor_2'].rolling(window=5).mean()
df_clean.dropna(inplace=True)


# 6. PREPARE LSTM INPUT WITH SEQUENCES


def create_sequences(data, labels, seq_length=20):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = labels.iloc[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_len = 20
X_train_seq, y_train_seq = create_sequences(X_clean[:1020], y_clean[:1020], seq_len)

# For validation, split from the same cleaned data (or use separate clean val data if available)
X_val_seq, y_val_seq = create_sequences(X_clean[1020:1120], y_clean[1020:1120], seq_len)


# 7. MODEL TRAINING


model = Sequential()
model.add(LSTM(64, input_shape=(seq_len, X_train_seq.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(1))  # Regression output

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.summary()

history = model.fit(X_train_seq, y_train_seq,
                    validation_data=(X_val_seq, y_val_seq),
                    epochs=30, batch_size=32,
                    callbacks=[early_stop])


# 8. EVALUATION

y_pred = model.predict(X_val_seq)
rmse = np.sqrt(mean_squared_error(y_val_seq, y_pred))
mae = mean_absolute_error(y_val_seq, y_pred)

print("RMSE:", rmse)
print("MAE:", mae)


# 9. SAVE MODEL

model.save("predictive_maintenance_lstm_model.h5")
print("Model saved successfully!")
