# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load your stock data (assuming it's in a CSV format with 'Date' and 'Close' columns)
# You can replace 'stock_data.csv' with your actual data file
data = pd.read_csv('stock_data.csv', parse_dates=['Date'], index_col='Date')

# Visualize the data
plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Stock Price')
plt.title('Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Split the data into training and test sets (e.g., 80% train, 20% test)
train_size = int(len(data) * 0.8)
train, test = data['Close'][:train_size], data['Close'][train_size:]

# Fit the ARIMA model
# (p, d, q) parameters can be chosen using ACF/PACF plots or grid search
model = ARIMA(train, order=(5, 1, 0))  # Here (5, 1, 0) is just an example
model_fit = model.fit()

# Make predictions
start = len(train)
end = len(train) + len(test) - 1
predictions = model_fit.predict(start=start, end=end, typ='levels')

# Plot the predictions against the actual test data
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.title('Stock Price Prediction using ARIMA')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Evaluate the model
mse = mean_squared_error(test, predictions)
print(f'Mean Squared Error: {mse}')
