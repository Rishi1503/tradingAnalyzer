import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,  accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import datetime
from sklearn.model_selection import RandomizedSearchCV

# Download historical data for the stock
stock_data = yf.download("AAPL", start="2023-01-01", end="2023-06-14")

print(stock_data)

# Create features and target variables

stock_data["Previous_Week"] = stock_data["Close"].shift(5)
stock_data["Next_Week"] = stock_data['Adj Close'].shift(-5)
stock_data['Next_Week'] = stock_data['Next_Week'].pct_change(5)
# Calculate moving averages
stock_data['MA_10'] = stock_data['Adj Close'].rolling(window=10).mean()
stock_data['MA_20'] = stock_data['Adj Close'].rolling(window=20).mean()
# stock_data['MA_100'] = stock_data['Adj Close'].rolling(window=100).mean()


# feature_names = []
# for n in [14, 30, 50, 200]:
#     delta = stock_data['Adj Close'].diff()
#     gain = delta.mask(delta < 0, 0)
#     loss = -delta.mask(delta > 0, 0)
#     avg_gain = gain.rolling(window=n).mean()
#     avg_loss = loss.rolling(window=n).mean()
#     rs = avg_gain / avg_loss
#     stock_data['rsi_' + str(n)] = 100 - (100 / (1 + rs))

for n in [7, 14, 30, 50]:
    stock_data['High-Low'] = stock_data['High'] - stock_data['Low']
    stock_data['High-PrevClose'] = abs(stock_data['High'] - stock_data['Close'].shift())
    stock_data['Low-PrevClose'] = abs(stock_data['Low'] - stock_data['Close'].shift())
    stock_data['TR'] = stock_data[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    stock_data['ATR' + str(n)] = stock_data['TR'].rolling(window=n).mean()


# Calculate Bollinger Bands
window_size = 20
num_std = 2

stock_data['std'] = stock_data['Adj Close'].rolling(window=window_size).std()
stock_data['upper_band'] = stock_data['MA_20'] + (num_std * stock_data['std'])
stock_data['lower_band'] = stock_data['MA_20'] - (num_std * stock_data['std'])

# Create features and target variables
stock_data['Previous_Close'] = stock_data['Close'].shift(1)
# stock_data2 = stock_data.copy()
# # print(stock_data2)
# stock_data2 = stock_data2.fillna(0)  # Replace NaN with 0
# print(stock_data2['Close'])

stock_data2 = stock_data.copy()
stock_data2 = stock_data2.fillna(stock_data2.mean())
# stock_data[['Open', 'High', 'Low', 'Previous_Close', 'Previous_Week', 'Volume', 'MA_20', 'MA_50', 'MA_200', 'rsi_14', 'rsi_100', 'rsi_200']].values
stock_data.dropna(inplace=True)
# stock_data = stock_data.fillna(stock_data.mean())
# x = stock_data[['Open', 'High', 'Low', 'Previous_Week', 'MA_20','MA_200']].values
# x = stock_data[['Open', 'Low', 'Close','MA_200','ATR200', 'upper_band', 'High']].values
x = stock_data[['MA_20','Open','High','Close', 'upper_band', 'lower_band', 'ATR7', 'ATR14', 'ATR30', 'ATR50']].values
# x2 = stock_data2[['Open', 'Low', 'Close','MA_200','ATR200', 'upper_band', 'High']].values
# x = stock_data[['MA_20', 'MA_50', 'MA_200', 'upper_band', 'lower_band']].values
# x = stock_data[['MA_20','MA_200','rsi_200']].values
y = stock_data['Next_Week'].dropna()

# x2 = stock_data[['Open', 'High', 'Low', 'Previous_Close', 'Previous_Week', 'Volume', 'MA_20', 'MA_50', 'MA_200', 'rsi_14', 'rsi_100', 'rsi_200']].values
# y2 = stock_data['Close']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

x2_train, x2_test, y2_train, y2_test = train_test_split(x, y, test_size=0.2, random_state=0)

# grid_rf = {
# 'n_estimators': [20, 50, 100, 500, 1000],  
# 'max_depth': np.arange(1, 15, 1),  
# 'min_samples_split': [2, 10, 9], 
# 'min_samples_leaf': np.arange(1, 15, 2, dtype=int),  
# 'bootstrap': [True, False], 
# 'random_state': [1, 2, 30, 42],
# }

# model = RandomForestRegressor()

# rscv = RandomizedSearchCV(estimator=model, param_distributions=grid_rf, cv=3, n_jobs=-1, verbose=2, n_iter=200)
# rscv_fit = rscv.fit(x_train, y_train)
# best_parameters = rscv_fit.best_params_
# print(best_parameters)
# # Create and train the Random Forest regressor
# rf_regressor =  RandomForestRegressor(n_estimators= best_parameters['n_estimators'], 
#                               random_state=best_parameters['random_state'], 
#                               min_samples_split=best_parameters['min_samples_split'], 
#                               min_samples_leaf=best_parameters['min_samples_leaf'], 
#                               max_depth=best_parameters['max_depth'], 
#                               bootstrap=best_parameters['bootstrap'])

rf_regressor =  RandomForestRegressor(n_estimators= 500, 
                              random_state=30, 
                              min_samples_split=2, 
                              min_samples_leaf=3, 
                              max_depth=9, 
                              bootstrap=True)
rf_regressor.fit(x_train, y_train)

# Make predictions on the test set
y_pred = rf_regressor.predict(x_test)
next_week_percentage_change = y_pred[-1]

print("Next week's percentage change:", next_week_percentage_change)

print("Mean Absolute Error:", round(metrics.mean_absolute_error(y_test, y_pred), 4))
print("Mean Squared Error:", round(metrics.mean_squared_error(y_test, y_pred), 4))
print("Root Mean Squared Error:", round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 4))
print("(R^2) Score:", round(metrics.r2_score(y_test, y_pred), 4))

importances = rf_regressor.feature_importances_
sorted_index = np.argsort(importances)[::-1]
x_values = range(len(importances))
labels = np.array(['MA_20','Open','High','Close', 'upper_band', 'lower_band', 'ATR7', 'ATR14', 'ATR30', 'ATR50'])[sorted_index]
plt.bar(x_values, importances[sorted_index], tick_label=labels)
plt.xticks(rotation=90)
plt.show()

y_pred_series = pd.Series(y_pred, index=y_test.index)
x_pred_series = pd.Series(y.values , index=y.index)
y_pred_series.plot(label='Predicted')
x_pred_series.plot(label='Actual')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Plot the actual and predicted prices


