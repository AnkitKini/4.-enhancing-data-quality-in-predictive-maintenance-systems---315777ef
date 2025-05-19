
# Predictive Maintenance Using LSTM

This project is a complete pipeline for predictive maintenance using a Long Short-Term Memory (LSTM) neural network. It predicts the Remaining Useful Life (RUL) of machines based on sensor data using the NASA CMAPSS dataset and a separate validation dataset.

## Project Phases

### 1. Environment Setup
- Installs necessary libraries: `pyod`, `fancyimpute`, `tensorflow`, `sklearn`, `matplotlib`, etc.

### 2. Data Loading
- Loads NASA's CMAPSS dataset and a custom validation dataset.
- Applies column names and handles missing data.

### 3. Preprocessing
- Computes RUL labels for the training dataset.
- Selects relevant features.
- Normalizes both datasets using `MinMaxScaler`.

### 4. Anomaly Detection
- Applies `IsolationForest` to filter outliers from the training data.

### 5. Feature Engineering
- Adds rolling statistics to enhance temporal features.

### 6. Prepare LSTM Input
- Converts flat input data into sequences (e.g., 20 time steps) for LSTM processing.

### 7. Model Training
- Defines a stacked LSTM model with dropout.
- Compiles and fits the model using training sequences.
- Implements early stopping.

### 8. Evaluation
- Computes RMSE and MAE on validation data.
- Saves model as `.h5` file.

### 9. Testing Phase
- Loads the saved model and recompiles it.
- Processes new data into sequences.
- Makes predictions on test samples.
- Outputs predicted RUL values.

### 10. Deployment Phase (Flask App)
- Implements a Flask API to serve the model for real-time prediction.
- Accepts CSV uploads, processes the file, and returns RUL predictions.

## Running in Google Colab

1. Upload your saved model (`.h5`) and Flask app script (`app.py`).
2. Run Flask with `!flask run --host=0.0.0.0 --port=5000` or using `flask_ngrok`.
3. Use public URL to test predictions with POST requests.

## Example Inputs
- Validation/test data should include columns like:
  - `Air temperature [K]`
  - `Process temperature [K]`
  - `Rotational speed [rpm]`
  - `Torque [Nm]`
  - `Tool wear [min]`

## File Structure

```
.
├── train_FD001.txt                  # NASA CMAPSS training data
├── predictive_maintenance.csv      # Validation/test dataset
├── predictive_maintenance_lstm_model.h5  # Trained LSTM model
├── app.py                          # Flask app script
└── README.md                       # Project documentation
```
