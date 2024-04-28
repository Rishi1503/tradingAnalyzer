end_date = "2023-07-28"
df = yf.download("AAPL", start="2022-11-01", end=end_date)

hma_period = 20


df['macd'], df['signal'], df['hist'] = ta.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['ema'] = ta.EMA(df['Close'], timeperiod=20)
df['ema_long'] = ta.EMA(df['Close'], timeperiod=40)
df['ema_short'] = ta.EMA(df['Close'], timeperiod=20)
df['hma'] = ta.WMA(ta.WMA(df['Close'], timeperiod=hma_period), timeperiod=int(np.sqrt(hma_period)))
# Calculate Relative Strength Index (RSI)
df['rsi'] = ta.RSI(df['Close'], timeperiod=14)
# Calculate Bollinger Bands
df['upper'], df['middle'], df['lower'] = ta.BBANDS(df['Close'], timeperiod=20)
df['bb_pct'] = (df['Close'] - df['middle']) / df['middle']  # Bollinger Band percentage
# Calculate On-Balance Volume (OBV)
df['obv'] = ta.OBV(df['Close'], df['Volume'])
# Calculate Average True Range (ATR)
df['atr_slow'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=30)
df['atr_fast'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=10)
# Calculate Williams %R
df['williams_r'] = ta.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
df['macd_signal'] = df['macd']/df['signal']
df['adx'] = ta.ADX(df['High'], df['Low'], df['Close'], timeperiod=20)
df['stoch_slowk'], df['stoch_slowd'] = ta.STOCH(df['High'], df['Low'], df['Close'],
                                                fastk_period=5, slowk_period=20, slowk_matype=0,
                                                slowd_period=20, slowd_matype=0)



df.dropna(inplace=True)
# Indexing Batches
train_df = df.sort_values(by=['Date']).copy()

# List of considered Features
FEATURES = [
    #'macd', 'upper', 'ema','Close', 'rsi', 'middle', 'signal', 'atr_slow', 'adx'
              # 'ema_long', 'rsi','Close', 'macd', 'signal', 'upper', 'lower', 'atr_slow', 'adx'
            'macd', 'signal', 'rsi', 'Close','ema_long', 'ema_short', 'obv', 'upper', 'middle'
            #, 'Month', 'Year', 'Adj Close'
           ]

print('FEATURE LIST')
print([f for f in FEATURES])

# Create the dataset with features and filter the data to the list of FEATURES
data = pd.DataFrame(train_df)
data_filtered = data[FEATURES]

# We add a prediction column and set dummy values to prepare the data for scaling
data_filtered_ext = data_filtered.copy()
data_filtered_ext['Prediction'] = data_filtered_ext['Close']

# Print the tail of the dataframe
data_filtered_ext.tail()

# Get the number of rows in the data
nrows = data_filtered.shape[0]

# Convert the data to numpy values
np_data_unscaled = np.array(data_filtered)
np_data = np.reshape(np_data_unscaled, (nrows, -1))
print(np_data.shape)

# Transform the data by scaling each feature to a range between 0 and 1
scaler = MinMaxScaler()
np_data_scaled = scaler.fit_transform(np_data_unscaled)

# Creating a separate scaler that works on a single column for scaling predictions
scaler_pred = MinMaxScaler()
df_Close = pd.DataFrame(data_filtered_ext['Close'])
np_Close_scaled = scaler_pred.fit_transform(df_Close)

# Set the sequence length - this is the timeframe used to make a single prediction
sequence_length = 50

# Prediction Index
index_Close = data.columns.get_loc("Close")

# Split the training data into train and train data sets
# As a first step, we get the number of rows to train the model on 80% of the data
train_data_len = math.ceil(np_data_scaled.shape[0] * 0.8)

# Create the training and test data
train_data = np_data_scaled[0:train_data_len, :]
test_data = np_data_scaled[train_data_len - sequence_length:, :]

# The RNN needs data with the format of [samples, time steps, features]
# Here, we create N samples, sequence_length time steps per sample, and 6 features
def partition_dataset(sequence_length, data):
    x, y = [], []
    data_len = data.shape[0]
    for i in range(sequence_length, data_len):
        x.append(data[i-sequence_length:i,:]) #contains sequence_length values 0-sequence_length * columsn
        y.append(data[i, index_Close]) #contains the prediction values for validation,  for single-step prediction

    # Convert the x and y to numpy arrays
    x = np.array(x)
    y = np.array(y)
    return x, y

# Generate training data and test data
x_train, y_train = partition_dataset(sequence_length, train_data)
x_test, y_test = partition_dataset(sequence_length, test_data)

# # Print the shapes: the result is: (rows, training_sequence, features) (prediction value, )
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

# # Validate that the prediction value and the input match up
# # The last close price of the second input sample should equal the first prediction value
# print(x_train[1][sequence_length-1][index_Close])
# print(y_train[0])

# Configure the neural network model
model = Sequential()

# Model with n_neurons = inputshape Timestamps, each with x_train.shape[2] variables
n_neurons = x_train.shape[1] * x_train.shape[2]
print(n_neurons, x_train.shape[1], x_train.shape[2])
model.add(LSTM(n_neurons, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(LSTM(n_neurons, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Training the model
epochs = 29
batch_size = 16
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test)
                   )

# Get the predicted values
y_pred_scaled = model.predict(x_test)

# Unscale the predicted values
y_pred = scaler_pred.inverse_transform(y_pred_scaled)
y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))

# Mean Absolute Error (MAE)
MAE = mean_absolute_error(y_test_unscaled, y_pred)
print(f'Median Absolute Error (MAE): {np.round(MAE, 2)}')

# Mean Absolute Percentage Error (MAPE)
MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled))) * 100
print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')

# Median Absolute Percentage Error (MDAPE)
MDAPE = np.median((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled)) ) * 100
print(f'Median Absolute Percentage Error (MDAPE): {np.round(MDAPE, 2)} %')

RMSE = np.sqrt(mean_squared_error(y_test_unscaled, y_pred))
print(f'Root Mean Squared Error (RMSE): {np.round(RMSE, 2)}')

R2 = r2_score(y_test_unscaled, y_pred)
print(f'R-squared: {np.round(R2, 2)}')