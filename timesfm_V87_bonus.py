# Student Name: Anas Uddin
# Student ID: 153215593
# Assigned Column: V87
# Bonus: TimesFM Forecast (Compare with ARIMA)
# --------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import PyTorch TimesFM
from timesfm import TimesFM_2p5_200M_torch as TimesFmTorch
from timesfm import ForecastConfig

# Load Data
data = pd.read_csv("Case_study.csv")
Y = data["V87"].values

# Split into training (1–100) and test (101–200)
Y_train = Y[:100]
Y_test = Y[100:]

# Pad Y_train to multiple of patch size
patch_size = 32  # default patch size for TimesFM_2p5_200M_torch
pad_len = (patch_size - (len(Y_train) % patch_size)) % patch_size
Y_train_padded = np.concatenate([Y_train, np.zeros(pad_len)])

# Initialize TimesFM Model
model = TimesFmTorch()

# Set max_horizon = 100
forecast_config = ForecastConfig(max_horizon=100)

# Compile the model
model.compile(forecast_config=forecast_config)

# Zero-Shot Forecast
forecast_zero = model.forecast(inputs=[Y_train_padded], horizon=100)
pred_zero = forecast_zero[0][: len(Y_test)]  # Trim extra padded points

# Compute MSE
mse_zero = np.mean((Y_test - pred_zero) ** 2)
print("Zero-shot MSE:", mse_zero)

# Plot Actual vs Forecast
plt.figure(figsize=(10, 6))

# Actual series
plt.plot(range(200), Y, label="Actual", color="black")

# Zero-shot forecast (flattened)
plt.plot(
    range(100, 200),
    pred_zero.flatten(),
    label="TimesFM Forecast (Zero-shot)",
    color="blue",
)

plt.xlabel("Time")
plt.ylabel("Y")
plt.title("TimesFM Forecast vs Actual Series (h = 100)")
plt.legend()
plt.show()
