# %% [markdown]
# Created by: Joseph Crouson
# Version: 0.1
# Date: 12/26/2020
# Last Modified:
# Purpose: Build, Train, Evaluate a DNN on the data

# %%
# Imports and Functions
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections.abc import Sequence
import pandas as pd
import numpy as np
import pickle
import time
import random

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping


# Code reused from CSC 180
# Convert a Pandas dataframe to the X, y inputs that TensorFlow needs


def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.
    target_type = df[target].dtypes
    target_type = target_type[0] if isinstance(
        target_type, Sequence) else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df[result].values.astype(np.float32), dummies.values.astype(np.float32)
    else:
        # Regression
        return df[result].values.astype(np.float32), df[target].values.astype(np.float32)


# Encode text values to indexes(i.e. [1],[2],[3] for red,green,blue).
def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_


# %%
# Get Data, Shuffle, Split
pickle_in = open(os.path.dirname(os.path.dirname(__file__)) +
                 r'\Data\PriceHistory\NQ_Data_ProperRecords_Normalized.pickle', 'rb')
priceData = pickle.load(pickle_in)
pickle_in.close()
del(pickle_in)

priceData.drop(priceData.iloc[:, 8:-1], inplace=True, axis=1)
priceData.drop(columns=['Open', 'Open_1_CandleAgo'])

# Shuffle Data
buys = priceData[priceData['Action'] == 'Buy']
sells = priceData[priceData['Action'] == 'Sell']
waits = priceData[priceData['Action'] == 'Wait']
sampleSize = min(buys.shape[0], sells.shape[0], waits.shape[0])
buys = buys.tail(sampleSize)
sells = sells.tail(sampleSize)
waits = waits.tail(sampleSize)
priceData = buys.append(sells, ignore_index=True)
priceData = priceData.append(waits, ignore_index=True)
priceData = priceData.sample(frac=1).reset_index(drop=True)
del(sampleSize)
del(buys)
del(sells)
del(waits)

# Split into X, y
actions = encode_text_index(priceData, 'Action')
X, y = to_xy(priceData, 'Action')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.13)


# %%
# Create the model

model = Sequential()

model.add(Dense(name='Input_Dense_40', units=20, activation='relu', input_dim=X.shape[1], kernel_regularizer=regularizers.l2(
    0.01), activity_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
model.add(Dropout(0.2))
BatchNormalization(axis=1)

# model.add(Dense(name='Hidden1_Dense_40', units=40, activation='relu', kernel_regularizer=regularizers.l2(
#    0.01), activity_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
# model.add(Dropout(0.2))
# BatchNormalization(axis=1)

# model.add(Dense(name='Hidden2_Dense_20', units=20, activation='relu', kernel_regularizer=regularizers.l2(
#    0.01), activity_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
# model.add(Dropout(0.2))
# BatchNormalization(axis=1)

model.add(Dense(name='Hidden3_Dense_10', units=10, activation='relu', kernel_regularizer=regularizers.l2(
    0.01), activity_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
model.add(Dropout(0.2))
BatchNormalization(axis=1)

model.add(Dense(name='Output_Dense_3', units=3, activation='softmax', kernel_regularizer=regularizers.l2(
    0.01), activity_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l2(0.01)))

# %%
model.summary()

# %%
# Compile and Fit

model.compile(loss='categorical_crossentropy', optimizer='sgd',
              metrics=['accuracy'])

#tensorBoard = TensorBoard(log_dir="logs/{}".format("Attempt1"))
# monitor = EarlyStopping(
#   monitor='val_loss', min_delta=1e-3, patience=5, verbose=0, mode='auto')
# unique file name that will include the epoch and the validation acc for that epoch
#filepath = "DNN_Final-{epoch:02d}-{categorical_accuracy:.3f}"
# checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='categorical_accuracy',
# verbose=0, save_best_only=True, mode='max'))  # saves only the best ones

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=1, epochs=10,
                    batch_size=6, shuffle=True)

# %%
