# 
# From: https://towardsdatascience.com/time-series-of-price-anomaly-detection-with-lstm-11a12ba4f6d9
# Notebook origins: https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Timeseries%20anomaly%20detection%20using%20LSTM%20Autoencoder%20JNJ.ipynb
#
# this is a little sketchy on the details, the sequence one seems more robust, or I understand it better. Also less complete.
#
# For multi variate: https://towardsdatascience.com/lstm-autoencoder-for-anomaly-detection-e1f4f2ee7ccf
# has a great follow up on REST and containerising: https://medium.com/swlh/containerized-ai-for-anomaly-detection-eb3e08225235# 
# in particular https://github.com/BLarzalere/LSTM-Autoencoder-for-Anomaly-Detection 
#



from tensorflow import keras
from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import tensorflow as tf
import pandas as pd

np.random.seed(1)
tf.random.set_seed(1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

print('Tensorflow version: ', tf.__version__)

#
# load the data
#
df = pd.read_csv('timeseries-data.csv')
print(df.head())

df = df[['Date', 'Close']]
print(df.dtypes)

# correct the types
df['Date'] = pd.to_datetime(df['Date'])
print(df.dtypes)



#
# Explore the data
#

print(df.describe())
print(df['Date'].min(), df['Date'].max())

# graph the timeseries
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close price'))
fig.update_layout(showlegend=True, title='Johnson and Johnson Stock Price 1985-2020')
fig.show()

# split the data - by date in this case
train, test = df.loc[df['Date'] <= '2013-09-03'], df.loc[df['Date'] > '2013-09-03']
print(train.tail())
print(test.head())

print(train.shape,test.shape)

# scale the data
scaler = StandardScaler()
scaler = scaler.fit(train[['Close']])
df.loc[df['Date'] <= '2013-09-03', 'Close'] = scaler.transform(train[['Close']])
df.loc[df['Date'] > '2013-09-03', 'Close'] = scaler.transform(test[['Close']])

train, test = df.loc[df['Date'] <= '2013-09-03'], df.loc[df['Date'] > '2013-09-03']
print(train.tail())
print(test.head())



TIME_STEPS=30

def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        Xs.append(X.iloc[i:(i+time_steps)].values)
        ys.append(y.iloc[i+time_steps])
    
    return np.array(Xs), np.array(ys)

X_train, y_train = create_sequences(train[['Close']], train['Close'])
X_test, y_test = create_sequences(test[['Close']], test['Close'])

print(f'Training shape: {X_train.shape}')
print(f'Testing shape: {X_test.shape}')



#
# Build the model
#



model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(rate=0.2))
model.add(RepeatVector(X_train.shape[1]))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(TimeDistributed(Dense(X_train.shape[2])))
model.compile(optimizer='adam', loss='mae')
print(model.summary())


# 
# Train the model
#

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')], shuffle=False)
print(model.evaluate(X_test, y_test))

# plot some stuff about it
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()




# 
# Determine anomalies
#

# Find MAE loss on the training data.
# Make the max MAE loss value in the training data as the reconstruction error threshold.
# If the reconstruction loss for a data point in the test set is greater than this reconstruction error threshold value then we will label this data point as an anomaly.

X_train_pred = model.predict(X_train, verbose=0)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

plt.hist(train_mae_loss, bins=50)
plt.xlabel('Train MAE loss')
plt.ylabel('Number of Samples')

threshold = np.max(train_mae_loss)
print(f'Reconstruction error threshold: {threshold}')

plt.show()

# Check it out on the test data (training data was a dry run)
X_test_pred = model.predict(X_test, verbose=0)
test_mae_loss = np.mean(np.abs(X_test_pred-X_test), axis=1)

plt.hist(test_mae_loss, bins=50)
plt.xlabel('Test MAE loss')
plt.ylabel('Number of samples')
plt.show()

# show threshold 
test_score_df = pd.DataFrame(test[TIME_STEPS:])
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = threshold
test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
test_score_df['Close'] = test[TIME_STEPS:]['Close']

fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df['Date'], y=test_score_df['loss'], name='Test loss'))
fig.add_trace(go.Scatter(x=test_score_df['Date'], y=test_score_df['threshold'], name='Threshold'))
fig.update_layout(showlegend=True, title='Test loss vs. Threshold')
fig.show()


anomalies = test_score_df.loc[test_score_df['anomaly'] == True]
print(anomalies.shape)

# 
# Visualise anomalies
# 

fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df['Date'], y=scaler.inverse_transform(test_score_df['Close']), name='Close price'))
fig.add_trace(go.Scatter(x=anomalies['Date'], y=scaler.inverse_transform(anomalies['Close']), mode='markers', name='Anomaly'))
fig.update_layout(showlegend=True, title='Detected anomalies')
fig.show()