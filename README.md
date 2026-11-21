# Ex.No: 6 – Time Series Forecasting using Nile Dataset

## AIM

To develop a Python program for time series forecasting using the Nile River dataset by applying Holt's Exponential Smoothing method for testing and ARIMA model for future prediction.

## ALGORITHM

1. Import the required Python libraries such as Pandas, NumPy, Matplotlib, Scikit-learn, and Statsmodels.
2. Load the `Nile.csv` dataset and remove unnecessary columns.
3. Rename the `time` column as `Year` and convert the columns into numeric format.
4. Set the Year column as the index and display the first five records and dataset information.
5. Plot the original Nile River flow data.
6. Apply RobustScaler to scale the dataset and plot the scaled data.
7. Perform time series decomposition using the additive model with a period of 10.
8. Split the dataset into training and testing data using an 80:20 ratio.
9. Apply RobustScaler to the training and testing data.
10. Build the Holt's Exponential Smoothing model with an additive trend and no seasonal component.
11. Forecast the test data and convert the predicted values back to the original scale.
12. Evaluate the model using MAE, MSE, RMSE, and MAPE.
13. Build an ARIMA(1,1,0) model using the entire dataset.
14. Forecast the future 10 years of Nile River flow.
15. Plot the final predictions along with the 95% confidence interval.
16. Display the predicted values and conclude the experiment.

## PROGRAM

```python
# Ex.No: 6 - Time Series Forecasting using Nile Dataset

# --------------------------------------------------
# 1. IMPORT LIBRARIES
# --------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------------
# 2. LOAD DATASET AND DATA EXPLORATION
# --------------------------------------------------

data = pd.read_csv("Nile.csv")

# Remove unnecessary column
data = data.drop(columns=['Unnamed: 0'], errors='ignore')

# Rename column
data = data.rename(columns={'time': 'Year'})

# Convert columns to numeric
data['Year'] = pd.to_numeric(data['Year'])
data['Nile'] = pd.to_numeric(data['Nile'])

# Set Year as index
data = data.set_index('Year')

print("First 5 Rows:")
print(data.head())

print("\nDataset Information:")
print(data.info())


# --------------------------------------------------
# 3. PLOT ORIGINAL DATA
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(data.index, data['Nile'])

plt.title("Nile River Flow")
plt.xlabel("Year")
plt.ylabel("River Flow")
plt.grid()
plt.show()


# --------------------------------------------------
# 4. SCALE DATA USING ROBUST SCALER
# --------------------------------------------------

scaler_plot = RobustScaler()

scaled_values = scaler_plot.fit_transform(
    data[['Nile']]
).flatten()

scaled_data = pd.Series(
    scaled_values,
    index=data.index
)

plt.figure(figsize=(10, 5))

plt.plot(scaled_data)

plt.title("Scaled Nile River Flow using RobustScaler")
plt.xlabel("Year")
plt.ylabel("Scaled Values")
plt.grid()
plt.show()


# --------------------------------------------------
# 5. TIME SERIES DECOMPOSITION
# --------------------------------------------------

# Period 10 is used for visualization only.
# Nile dataset is yearly and has no fixed seasonal cycle.

decomposition = seasonal_decompose(
    data['Nile'],
    model='additive',
    period=10
)

decomposition.plot()

plt.suptitle(
    "Nile River Flow Decomposition",
    y=1.02
)

plt.show()


# --------------------------------------------------
# 6. TRAIN-TEST SPLIT (80:20)
# --------------------------------------------------

split = int(len(data) * 0.8)

train_data = data['Nile'].iloc[:split]
test_data = data['Nile'].iloc[split:]

print("\nTraining Data:")
print(train_data.index.min(), "-", train_data.index.max())

print("\nTesting Data:")
print(test_data.index.min(), "-", test_data.index.max())


# --------------------------------------------------
# 7. ROBUST SCALING
# --------------------------------------------------

scaler = RobustScaler()

train_scaled = scaler.fit_transform(
    train_data.values.reshape(-1, 1)
).flatten()

test_scaled = scaler.transform(
    test_data.values.reshape(-1, 1)
).flatten()


# --------------------------------------------------
# 8. HOLT-WINTERS MODEL
# --------------------------------------------------

model = ExponentialSmoothing(
    train_scaled,
    trend='add',
    seasonal=None,
    initialization_method='estimated'
).fit()


# --------------------------------------------------
# 9. TEST PREDICTIONS
# --------------------------------------------------

test_predictions_scaled = model.forecast(
    steps=len(test_data)
)

# Convert scaled predictions back to original values
test_predictions = scaler.inverse_transform(
    test_predictions_scaled.reshape(-1, 1)
).flatten()


# --------------------------------------------------
# 10. VISUAL EVALUATION
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    train_data.index,
    train_data,
    label='Train Data'
)

plt.plot(
    test_data.index,
    test_predictions,
    label='Test Predictions'
)

plt.plot(
    test_data.index,
    test_data,
    label='Actual Test Data'
)

plt.title("Visual Evaluation - Holt's Method")
plt.xlabel("Year")
plt.ylabel("River Flow")

plt.legend()
plt.grid()
plt.show()


# --------------------------------------------------
# 11. MODEL PERFORMANCE METRICS
# --------------------------------------------------

mae = mean_absolute_error(
    test_data,
    test_predictions
)

mse = mean_squared_error(
    test_data,
    test_predictions
)

rmse = np.sqrt(mse)

mape = np.mean(
    np.abs(
        (test_data.values - test_predictions)
        / test_data.values
    )
) * 100

print("\nHolt Model Performance Metrics")

print("MAE:", round(mae, 4))
print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("MAPE:", round(mape, 2), "%")

print("\nStandard Deviation and Mean:")

print("Standard Deviation:", data['Nile'].std())
print("Mean:", data['Nile'].mean())


# --------------------------------------------------
# 12. FINAL MODEL USING ARIMA(1,1,0)
# --------------------------------------------------

# Create a proper yearly time index
data.index = pd.date_range(
    start='1871',
    periods=len(data),
    freq='YS'
)

# Train final model
final_model = ARIMA(
    data['Nile'],
    order=(1, 1, 0)
).fit()


# --------------------------------------------------
# 13. FORECAST FUTURE 10 YEARS
# --------------------------------------------------

forecast_result = final_model.get_forecast(
    steps=10
)

# Predicted values
final_predictions = forecast_result.predicted_mean

# Confidence interval
confidence_interval = forecast_result.conf_int()

# Future year index
future_years = pd.date_range(
    start='1971',
    periods=10,
    freq='YS'
)

# Convert forecast index to years
final_predictions.index = future_years.year

confidence_interval.index = future_years.year

print("\nFuture 10-Year Predictions:")
print(final_predictions)


# --------------------------------------------------
# 14. PLOT FINAL FUTURE PREDICTION
# --------------------------------------------------

plt.figure(figsize=(12, 6))

# Original data
plt.plot(
    data.index.year,
    data['Nile'],
    label='Original Data'
)

# Future predictions
plt.plot(
    final_predictions.index,
    final_predictions,
    marker='o',
    linestyle='--',
    label='Future Predictions'
)

# Confidence interval
plt.fill_between(
    final_predictions.index,
    confidence_interval.iloc[:, 0],
    confidence_interval.iloc[:, 1],
    alpha=0.2,
    label='95% Confidence Interval'
)

plt.title("Nile River Flow Prediction using ARIMA(1,1,0)")
plt.xlabel("Year")
plt.ylabel("River Flow")

plt.legend()
plt.grid()
plt.show()
```

## OUTPUT
<img width="1115" height="1198" alt="image" src="https://github.com/user-attachments/assets/a9afa73f-2760-4fa0-878c-853fe0fbcc91" />

<img width="932" height="1171" alt="image" src="https://github.com/user-attachments/assets/b6e4a857-7822-4303-b738-c4e9a3437713" />

<img width="1217" height="1036" alt="image" src="https://github.com/user-attachments/assets/5e5c33f6-91e8-48db-ab5e-097378da2b30" />

## RESULT

Thus, the time series forecasting program was successfully implemented using the Nile dataset. Holt's Exponential Smoothing method was used for testing and performance evaluation, and the ARIMA(1,1,0) model was used to forecast the Nile River flow for the next 10 years.
