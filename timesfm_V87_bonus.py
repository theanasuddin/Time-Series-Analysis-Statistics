# Student Name: Anas Uddin
# Student ID: 153215593
# Assigned Column: V87
# Bonus: TimesFM Forecast (Compare with ARIMA)
# --------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import timesfm

torch.set_float32_matmul_precision("high")

# Load Data
data = pd.read_csv("Case_study.csv")
Y = data["V87"].values

# Training = first 100, Test = next 100
Y_train = Y[:100]
Y_test = Y[100:]

# Load TimesFM model
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

# Compile model with config
model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)

# Zero-shot Forecast (h = 100)
point_forecast, quantile_forecast = model.forecast(
    horizon=100,
    inputs=[Y_train],
)

# Extract forecast for our single series
pred_zero = point_forecast[0]

# Compute MSE
mse_zero = np.mean((Y_test - pred_zero) ** 2)
print("TimesFM Zero-shot MSE:", mse_zero)

# Plot Actual vs TimesFM Forecast
plt.figure(figsize=(10, 6))

# Actual full series
plt.plot(range(200), Y, label="Actual", color="black")

# TimesFM forecast (100 → 199)
plt.plot(
    range(100, 200),
    pred_zero,
    label="TimesFM Forecast (Zero-shot)",
    color="blue",
    linewidth=2,
)

plt.xlabel("Time")
plt.ylabel("Y")
plt.title("TimesFM Zero-Shot Forecast vs Actual (h = 100)")
plt.legend()
plt.grid(True)
plt.show()
