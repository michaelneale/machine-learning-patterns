# based on https://github.com/sergiovirahonda/TweetsSentimentAnalysis/blob/main/TweetsSentimentPredictions.ipynb
# and https://github.com/sergiovirahonda/TweetsSentimentAnalysis/blob/main/ModelDeployment.ipynb for training code


import argparse
import os
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# hyper params
epochs     = 10
max_words  = 5000
max_len    = 200

data = pd.read_csv('./dataset.csv',sep=',')

labels = data['sentiment'].values
labels = tf.keras.utils.to_categorical(labels, 3, dtype="float32")
features = data['selected_text'].values
X = []
for i in range(len(features)):
    X.append(str(features[i]))

#Tokenizing data and making them sequences
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
features = pad_sequences(sequences, maxlen=max_len)

#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(features,labels, random_state=0)
print (len(X_train),len(X_test),len(y_train),len(y_test))

# Building the model
model = Sequential()
model.add(layers.Embedding(max_words, 40, input_length=max_len))
model.add(layers.Bidirectional(layers.LSTM(20,dropout=0.6)))
model.add(layers.Dense(3,activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=10,
        verbose=1,
    )
]
history = model.fit(X_train, y_train, epochs=epochs,validation_data=(X_test, y_test), callbacks=callbacks)

#Validating model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('Model accuracy: ',test_acc)


sentiment = ['Neutral','Negative','Positive']

sequence = tokenizer.texts_to_sequences(['this experience has been the worst , want my money back'])
test = pad_sequences(sequence, maxlen=max_len)
print(sentiment[np.around(model.predict(test), decimals=0).argmax(axis=1)[0]])

sequence = tokenizer.texts_to_sequences(['do not think this will work , we talked about this and disagreed'])
test = pad_sequences(sequence, maxlen=max_len)
print(sentiment[np.around(model.predict(test), decimals=0).argmax(axis=1)[0]])

sequence = tokenizer.texts_to_sequences(['looks good to me , thanks'])
test = pad_sequences(sequence, maxlen=max_len)
print(sentiment[np.around(model.predict(test), decimals=0).argmax(axis=1)[0]])

sequence = tokenizer.texts_to_sequences(['do not merge this, this is terrible and will break everything'])
test = pad_sequences(sequence, maxlen=max_len)
print(sentiment[np.around(model.predict(test), decimals=0).argmax(axis=1)[0]])
