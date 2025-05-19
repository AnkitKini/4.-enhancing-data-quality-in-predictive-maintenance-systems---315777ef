from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = load_model("model/predictive_maintenance_lstm_model.h5", compile=False)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Define expected features
expected_features = ['Air temperature [K]', 'Process temperature [K]',
                     'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

# Define scaler (simulate for this deployment)
scaler = MinMaxScaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = []
        for feature in expected_features:
            value = float(request.form[feature])
            input_data.append(value)

        # Simulate historical data for 20 timesteps
        input_sequence = np.tile(input_data, (20, 1))  # shape: (20, features)
        input_scaled = scaler.fit_transform(input_sequence)  # scale (placeholder fit)
        input_scaled = np.expand_dims(input_scaled, axis=0)  # shape: (1, 20, features)

        prediction = model.predict(input_scaled)[0][0]
        return render_template('index.html', prediction_text=f'Predicted RUL: {prediction:.2f} cycles')

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
