# Student Name: Anas Uddin
# Student ID: 153215593
# Assigned Column: V87
# ------------------------------------------

# Step 1: Preliminary analysis of orders ---
# 1. Plot the times series(Yt)

# Load required libraries
library(readr)

# Load the dataset
data <- read_csv("Case_study.csv")

# Extract assigned series
Yt <- data$V87

# Plot the time series
plot(
  Yt,
  type = "l",
  col = "blue",
  main = "Time Series Plot of Yt (V87)",
  xlab = "Time",
  ylab = "Yt"
)

# Step 1: Preliminary analysis of orders-- -
# 2. Analysis of d
# (a) ACF of Yt, ∇Yt, and∇²Yt

# ACF of the original series
acf(Yt, main = "ACF of Yt")

# First difference
dYt1 <- diff(Yt)
acf(dYt1, main = "ACF of First Differenced Series (∇Yt)")

# Second difference
dYt2 <- diff(dYt1)
acf(dYt2, main = "ACF of Second Differenced Series (∇²Yt)")

# Step 1: Preliminary analysis of orders-- -
# 2. Analysis of d
# (b) Augmented Dickey-Fuller (ADF) test

library(tseries)

# ADF test on original series
adf.test(Yt)

# ADF test on first differenced series
adf.test(dYt1)

# ADF test on second differenced series
adf.test(dYt2)

# Step 1: Preliminary analysis of orders-- -
# 3. Analysis of autoregressive order p and moving average order q

# ACF and PACF of first differenced series
acf(dYt1, main = "ACF of ∇Yt")
pacf(dYt1, main = "PACF of ∇Yt")

# Step 2: Estimation and selection of ARIMA Models ---
# 1. Model estimation and information criteria

library(readr)

# Load data
data <- read_csv("Case_study.csv")

# Extract assigned series
Yt <- data$V87

# Use first 100 observations as training data
Y_train <- Yt[1:100]

# Differencing order (from Step 1)
d <- 1

# Maximum orders
pmax <- 4
qmax <- 4

# Create empty storage
results <- data.frame(
  p = integer(),
  q = integer(),
  AIC = numeric(),
  BIC = numeric()
)

# Loop over all combinations 1 ≤ p ≤ 4, 1 ≤ q ≤ 4
for (p in 1:pmax) {
  for (q in 1:qmax) {
    model <- arima(Y_train, order = c(p, d, q))
    
    results <- rbind(results, data.frame(
      p = p,
      q = q,
      AIC = AIC(model),
      BIC = BIC(model)
    ))
  }
}

# Display results
results

# Step 2: Estimation and selection of ARIMA Models ---
# 2. Selection of the best three specifications

# Order by AIC
results_AIC <- results[order(results$AIC), ]

# Order by BIC
results_BIC <- results[order(results$BIC), ]

# Best three according to AIC
best_AIC_models <- head(results_AIC, 3)

# Best three according to BIC
best_BIC_models <- head(results_BIC, 3)

best_AIC_models
best_BIC_models

# Step 2: Estimation and selection of ARIMA Models ---
# 3. Parameter estimates of the best three specifications

# Select three best unique models (based on AIC and BIC)
best_models <- unique(rbind(best_AIC_models[1:3, c("p", "q")], best_BIC_models[1:3, c("p", "q")]))

# Fit and display each model
for (i in 1:nrow(best_models)) {
  p_val <- best_models$p[i]
  q_val <- best_models$q[i]
  
  cat("\n==============================\n")
  cat("ARIMA(", p_val, ",1,", q_val, ")\n", sep = "")
  
  model <- arima(Y_train, order = c(p_val, 1, q_val))
  print(model)
}

# Step 3: Diagnostic Tests with In-Sample Data ---
# 1. LjungBox test (first 10 lags)

# Fit the three selected models again
model_411 <- arima(Y_train, order = c(4, 1, 1))
model_412 <- arima(Y_train, order = c(4, 1, 2))
model_413 <- arima(Y_train, order = c(4, 1, 3))

# Extract residuals
res_411 <- residuals(model_411)
res_412 <- residuals(model_412)
res_413 <- residuals(model_413)

# Ljung-Box test (lag = 10)
Box.test(res_411, lag = 10, type = "Ljung-Box")
Box.test(res_412, lag = 10, type = "Ljung-Box")
Box.test(res_413, lag = 10, type = "Ljung-Box")

# ACF and PACF of residuals
acf(res_411, main = "ACF Residuals ARIMA(4,1,1)")
pacf(res_411, main = "PACF Residuals ARIMA(4,1,1)")

acf(res_412, main = "ACF Residuals ARIMA(4,1,2)")
pacf(res_412, main = "PACF Residuals ARIMA(4,1,2)")

acf(res_413, main = "ACF Residuals ARIMA(4,1,3)")
pacf(res_413, main = "PACF Residuals ARIMA(4,1,3)")

# Step 3: Diagnostic Tests with In-Sample Data ---
# 2. Normality of residuals

# Histograms
hist(res_411, main = "Histogram Residuals ARIMA(4,1,1)")
hist(res_412, main = "Histogram Residuals ARIMA(4,1,2)")
hist(res_413, main = "Histogram Residuals ARIMA(4,1,3)")

# QQ plots
qqnorm(res_411)
qqline(res_411)
qqnorm(res_412)
qqline(res_412)
qqnorm(res_413)
qqline(res_413)

# Shapiro-Wilk test
shapiro.test(res_411)
shapiro.test(res_412)
shapiro.test(res_413)

# Step 3: Diagnostic Tests with In-Sample Data ---
# 4. Plot original series and fitted values

# Extract fitted values
fitted_411 <- Y_train - res_411

# Plot
plot(
  Y_train,
  type = "l",
  col = "black",
  main = "Original Series vs Fitted Values (ARIMA(4,1,1))",
  xlab = "Time",
  ylab = "Y"
)

lines(fitted_411, col = "red")
legend(
  "topleft",
  legend = c("Original", "Fitted"),
  col = c("black", "red"),
  lty = 1
)

# Step 4: Forecast with Out-of-Sample Data ---
# 1. Generate forecasts (h = 10, 25, 100)

# Preferred model from Step 3
final_model <- arima(Y_train, order = c(4, 1, 1))

# Forecast horizons
forecast_10  <- predict(final_model, n.ahead = 10)
forecast_25  <- predict(final_model, n.ahead = 25)
forecast_100 <- predict(final_model, n.ahead = 100)

# Extract forecast means and standard errors
pred_10  <- forecast_10$pred
se_10    <- forecast_10$se

pred_25  <- forecast_25$pred
se_25    <- forecast_25$se

pred_100 <- forecast_100$pred
se_100   <- forecast_100$se

# 95% Confidence Intervals
upper_10  <- pred_10  + 1.96 * se_10
lower_10  <- pred_10  - 1.96 * se_10

upper_25  <- pred_25  + 1.96 * se_25
lower_25  <- pred_25  - 1.96 * se_25

upper_100 <- pred_100 + 1.96 * se_100
lower_100 <- pred_100 - 1.96 * se_100

# Forecast plot (h = 10)
plot(
  Y,
  type = "l",
  col = "black",
  main = "Forecast (h = 10) using ARIMA(4,1,1)",
  xlab = "Time",
  ylab = "Y"
)

# Add forecast line
lines(101:110, pred_10, col = "blue", lwd = 2)

# Add 95% confidence interval
lines(101:110, upper_10, col = "red", lty = 2)
lines(101:110, lower_10, col = "red", lty = 2)

legend(
  "topleft",
  legend = c("Original", "Forecast", "95% CI"),
  col = c("black", "blue", "red"),
  lty = c(1, 1, 2)
)

# Forecast plot (h = 25)
plot(
  Y,
  type = "l",
  col = "black",
  main = "Forecast (h = 25) using ARIMA(4,1,1)",
  xlab = "Time",
  ylab = "Y"
)

# Add forecast line
lines(101:125, pred_25, col = "blue", lwd = 2)

# Add 95% confidence interval
lines(101:125, upper_25, col = "red", lty = 2)
lines(101:125, lower_25, col = "red", lty = 2)

legend(
  "topleft",
  legend = c("Original", "Forecast", "95% CI"),
  col = c("black", "blue", "red"),
  lty = c(1, 1, 2)
)

# Forecast plot (h = 100)
# Plot original series (all 200 observations)
plot(
  Y,
  type = "l",
  col = "black",
  main = "Forecast using ARIMA(4,1,1)",
  xlab = "Time",
  ylab = "Y"
)

# Add forecast (starting at 101)
lines(101:200, pred_100, col = "blue", lwd = 2)

# Add confidence intervals
lines(101:200, upper_100, col = "red", lty = 2)
lines(101:200, lower_100, col = "red", lty = 2)

legend(
  "topleft",
  legend = c("Original", "Forecast", "95% CI"),
  col = c("black", "blue", "red"),
  lty = c(1, 1, 2)
)

# Step 4: Forecast with Out-of-Sample Data ---
# 2. Mean Squared Error (MSE)

# Split the data
Y_test  <- Y[101:200]

# Compute forecast errors for 100-step forecast
errors <- Y_test - pred_100

# Mean Squared Error
MSE <- mean(errors^2)

MSE
