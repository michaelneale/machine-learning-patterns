# this is from https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/regression.ipynb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)



# Get data from https://archive.ics.uci.edu/ml/


url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)


dataset = raw_dataset.copy()
print(dataset.tail())                          

# Cleaning the data

dataset.isna().sum()
dataset = dataset.dropna()


# 
#The "Origin" column is really categorical, not numeric. So convert that to a one-hot:

# Note: You can set up the keras.Model to do this kind of transformation for you. That's beyond the scope of this tutorial. See the preprocessing layers or Loading CSV data tutorials for examples.
# https://render.githubusercontent.com/structured_data/preprocessing_layers.ipynb and https://render.githubusercontent.com/load_data/csv.ipynb

dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
print(dataset.tail())


# test/train split
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)


# take a look at the data
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
plt.show()

# Also look at the overall statistics, note how each feature covers a very different range:
print(train_dataset.describe().transpose())



# separate target out from the data that we will want to predict
train_features = train_dataset.copy()
test_features = test_dataset.copy()

# pop off the target so we don't leak it when training 
train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

# take a look at the dat to see if we need to normalise it: 
print(train_dataset.describe().transpose()[['mean', 'std']])

# you will see we do need to normalise the shizz out of it
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))

print(normalizer.mean.numpy())


first = np.array(train_features[:1])

# When the layer is called it returns the input data, with each feature independently normalized:
with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())




def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)

# Skipping ahead fo deep neural network regression: 


# Neural network structure: 
#   The normalization layer.
#   Two hidden, nonlinear, Dense layers using the relu nonlinearity.
#   A linear single-output layer.

def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

# prepare the model
dnn_model = build_and_compile_model(normalizer)
print(dnn_model.summary())

# train the model
history = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

# show the loss
plot_loss(history)
plt.show()



test_results = {}

test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)
print(pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T)


# try out making some predictions
test_predictions = dnn_model.predict(test_features).flatten()


# show it on X/Y to see how well things line up: 
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()


# show the error distribution: 
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')
plt.show()


# this is how you save and load the model: 
dnn_model.save('dnn_model')

reloaded = tf.keras.models.load_model('dnn_model')
test_results['reloaded'] = reloaded.evaluate(test_features, test_labels, verbose=0)
print(pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T)    




