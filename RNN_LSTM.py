# %%
# Import Data

from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers

import os
import time
from Data.DataFormat import process
import pandas as pd
import numpy as np
import datetime as dt
from collections import deque
import random
import pickle

START_TIME = time.time()
BUILD_NEW_MODELS = True
FULL_TEST = False

SYMBOL = 'NQ'
AGG_PERIOD = 3
FUTURE_PREDICTION_LEN = 1
PRICE_DELTA = 10
PREDICTION_SUPPORT_LEN = 2
#2018 or later
START_DATE = dt.date(2020, 1, 1)
WAIT_LABLE_ADDED = False
SIZE = 'S'


dateStr = START_DATE.strftime('%m.%d.%Y')
NAME = f'{SYMBOL}-AP{AGG_PERIOD}-FPL{FUTURE_PREDICTION_LEN}-D{PRICE_DELTA}-PSL{PREDICTION_SUPPORT_LEN}-SD-{dateStr}'
del(dateStr)


# Make sequences if they don't exist
sequentialData = []


def getSequences():

    if os.path.exists(os.path.dirname(__file__)+f'\\Data\\SequenceData\\{NAME}_SEQUENCES'):
        pickle_in = open(os.path.dirname(__file__) +
                         f'\\Data\\SequenceData\\{NAME}_SEQUENCES\\sequences', 'rb')
        sequentialData = pickle.load(pickle_in)
        pickle_in.close()
        print(f"\n\nRETURNING SAVED SEQUENCES OF: {NAME}\n\n")
        return sequentialData
    else:
        # FIRST PLACE THAT NEEDS TO BE OPTIMIZED IF POSSIBLE
        # Scale on %change for each column
        priceData = process(SYMBOL, AGG_PERIOD, START_DATE,
                            FUTURE_PREDICTION_LEN, PRICE_DELTA)
        actions = priceData['Action']
        dates = priceData['Date']
        priceData.drop(columns=['Action', 'Date',
                                'Time', 'Volume'], inplace=True)
        scaled_df = preprocessing.MinMaxScaler(
            feature_range=(-1, 1)).fit_transform(priceData.values)
        priceData = pd.DataFrame(
            scaled_df, columns=priceData.columns, index=priceData.index)
        priceData['Action'] = actions
        priceData['Date'] = dates

        print(f'GETTING NEW SEQUENCE DATA FOR: {NAME}')
        # Mark the start of a new day
        priceData['Date2'] = priceData['Date'].shift(-1)
        priceData['Check'] = priceData['Date'] != priceData['Date2']
        priceData.drop(columns=['Date2', 'Date'], inplace=True)

        sequentialData = []
        prev_candles = deque(maxlen=PREDICTION_SUPPORT_LEN)

        for index, i in enumerate(priceData.values):
            if priceData.values[index][-1] == True:
                prev_candles.append([n for n in i[:-2]])
                if len(prev_candles) == PREDICTION_SUPPORT_LEN:
                    sequentialData.append(
                        [np.array(prev_candles).astype(np.float32), i[-2]])
                    prev_candles.clear()
            else:
                prev_candles.append([n for n in i[:-2]])
                if len(prev_candles) == PREDICTION_SUPPORT_LEN:
                    sequentialData.append(
                        [np.array(prev_candles).astype(np.float32), i[-2]])
        os.mkdir(os.path.dirname(__file__) +
                 f'\\Data\\SequenceData\\{NAME}_SEQUENCES')
        pickle_out = open(os.path.dirname(
            __file__)+f'\\Data\\SequenceData\\{NAME}_SEQUENCES\\sequences', 'wb')
        pickle.dump(sequentialData, pickle_out)
        pickle_out.close()

        return sequentialData


sequentialData = getSequences()

# Balance
random.shuffle(sequentialData)

if WAIT_LABLE_ADDED:
    wait_lable = 'T'
else:
    wait_lable = 'F'

if not FULL_TEST:
    buys = []
    sells = []
    waits = []
    for seq, target in sequentialData:
        if target == 'Buy':
            buys.append([seq, target])
        elif target == 'Sell':
            sells.append([seq, target])
        elif target == 'Wait':
            waits.append([seq, target])

    lower = min(len(buys), len(sells), len(waits))

    buys = buys[:lower]
    sells = sells[:lower]
    waits = waits[:lower]

    if WAIT_LABLE_ADDED:
        sequentialData = buys + sells + waits
    else:
        sequentialData = buys + sells

    del(buys)
    del(sells)
    del(waits)
    del(seq)
    del(target)
    del(lower)

    random.shuffle(sequentialData)
elif not WAIT_LABLE_ADDED:
    buys = []
    sells = []
    waits = []
    for seq, target in sequentialData:
        if target == 'Buy':
            buys.append([seq, target])
        elif target == 'Sell':
            sells.append([seq, target])
        elif target == 'Wait':
            waits.append([seq, target])

    sequentialData = buys + sells

    del(buys)
    del(sells)
    del(waits)
    del(seq)
    del(target)

    random.shuffle(sequentialData)

# Prep for training

X = []
y = []

for seq, target in sequentialData:
    X.append(seq)
    y.append(target)

X = np.array(X)
del(seq)
del(target)

# y must be onehot encoded
yOneHot = pd.DataFrame(data=y, columns=['Action'])


def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)


def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_


yOneHotcp = yOneHot.copy()
actions = encode_text_index(yOneHot, 'Action')
encode_text_dummy(yOneHotcp, 'Action')

y = np.array(yOneHotcp.values).astype(np.float32)

del(sequentialData)
del(yOneHot)
del(yOneHotcp)


# Test and validation sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.35)


# %%
trainStart = time.time()
if BUILD_NEW_MODELS:
    # Build Model

    for i in range(1, 2):
        model = None
        if SIZE == 'L':
            model = Sequential()

            model.add(LSTM(80, input_shape=(
                X_train.shape[1:]), return_sequences=True, kernel_regularizer=regularizers.l2(
                0.01), activity_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
            model.add(Dropout(0.20))
            model.add(BatchNormalization())

            model.add(LSTM(60, input_shape=(
                X_train.shape[1:]), return_sequences=True, kernel_regularizer=regularizers.l2(
                0.01), activity_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
            model.add(Dropout(0.20))
            model.add(BatchNormalization())

            model.add(LSTM(40, input_shape=(
                X_train.shape[1:]), return_sequences=False, kernel_regularizer=regularizers.l2(
                0.01), activity_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
            model.add(Dropout(0.20))
            model.add(BatchNormalization())

            model.add(Dense(30, activation='relu', kernel_regularizer=regularizers.l2(
                0.01), activity_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
            model.add(Dropout(0.20))
            model.add(BatchNormalization())

            model.add(Dense(20, activation='relu', kernel_regularizer=regularizers.l2(
                0.01), activity_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
            model.add(Dropout(0.20))
            model.add(BatchNormalization())

            model.add(Dense(10, activation='relu', kernel_regularizer=regularizers.l2(
                0.01), activity_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
            model.add(Dropout(0.20))
            model.add(BatchNormalization())

            model.add(Dense(y_train.shape[1], activation='softmax'))

        elif SIZE == 'M':
            model = Sequential()

            model.add(LSTM(60, input_shape=(
                X_train.shape[1:]), return_sequences=True, kernel_regularizer=regularizers.l2(
                0.01), activity_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
            model.add(Dropout(0.20))
            model.add(BatchNormalization())

            model.add(LSTM(40, input_shape=(
                X_train.shape[1:]), return_sequences=True, kernel_regularizer=regularizers.l2(
                0.01), activity_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
            model.add(Dropout(0.20))
            model.add(BatchNormalization())

            model.add(LSTM(20, input_shape=(
                X_train.shape[1:]), return_sequences=False, kernel_regularizer=regularizers.l2(
                0.01), activity_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
            model.add(Dropout(0.20))
            model.add(BatchNormalization())

            model.add(Dense(15, activation='relu', kernel_regularizer=regularizers.l2(
                0.01), activity_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
            model.add(Dropout(0.20))
            model.add(BatchNormalization())

            model.add(Dense(9, activation='relu', kernel_regularizer=regularizers.l2(
                0.01), activity_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
            model.add(Dropout(0.20))
            model.add(BatchNormalization())

            model.add(Dense(y_train.shape[1], activation='softmax'))

        elif SIZE == 'S':
            model = Sequential()

            model.add(LSTM(48, input_shape=(
                X_train.shape[1:]), return_sequences=True, kernel_regularizer=regularizers.l2(
                0.01), activity_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
            model.add(Dropout(0.20))
            model.add(BatchNormalization())

            model.add(LSTM(24, input_shape=(
                X_train.shape[1:]), return_sequences=False, kernel_regularizer=regularizers.l2(
                0.01), activity_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
            model.add(Dropout(0.20))
            model.add(BatchNormalization())

            model.add(Dense(12, activation='relu', kernel_regularizer=regularizers.l2(
                0.01), activity_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
            model.add(Dropout(0.20))
            model.add(BatchNormalization())

            model.add(Dense(6, activation='relu', kernel_regularizer=regularizers.l2(
                0.01), activity_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
            model.add(Dropout(0.20))
            model.add(BatchNormalization())

            model.add(Dense(y_train.shape[1], activation='softmax'))
        else:
            raise ValueError('No model size specified')

        now = dt.datetime.now().strftime('%m.%d.%Y_%H.%M.%S')
        for j in range(2):
            if j == 0:
                opt = Adam(lr=0.001, decay=1e-6)
                opt_Name = 'ADAM'
                epochs = 30
            else:
                opt = SGD(lr=1e-5, decay=1e-6)
                opt_Name = 'SGD'
                epochs = 30

            # Compile and Train
            model.compile(loss='categorical_crossentropy', optimizer=opt,
                          metrics=['accuracy'])
            # In a terminal, navigate to the folder that contains TBlogs
            # Command: tensorboard --logdir TBlogs/W{wait_lable}/{NAME}

            fileName = f"Data/ModelStorage/W{wait_lable}/{SIZE}/{NAME}/V{i}_{opt_Name}_BUILT_{now}.hdf5"
            tensorboard = TensorBoard(
                log_dir=f'TBlogs/W{wait_lable}/{SIZE}/{NAME}/V{i}_{opt_Name}_BUILT_{now}')
            checkpointer = ModelCheckpoint(filepath=fileName, monitor='val_accuracy',
                                           mode='max', verbose=0, save_best_only=True)  # save best model
            earlyStop = EarlyStopping(
                monitor='loss', min_delta=1e-3, patience=7, mode='min', restore_best_weights=True)
            print(f'\n\n\n\n\n')
            print(f'Model saved at {fileName}')
            print(
                f'Model logged at TBlogs/W{wait_lable}/{SIZE}/{NAME}/V{i}_{opt_Name}_BUILT_{now}')
            print(
                f'Track model on TensorBoard: tensorboard --logdir TBlogs/W{wait_lable}/{SIZE}/{NAME}')
            print(f'\n\n\n\n\n')
            model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=1,
                      epochs=epochs, batch_size=2, shuffle=True, callbacks=[tensorboard, checkpointer, earlyStop])


print(f'\n\n\nTraining Time: {(time.time() - trainStart)//60}\n')

# %%
# Get the best model
print()
print('Loading the best model')
print()

winner = None
winnerScore = None
modelPath = None
models = os.listdir(os.path.dirname(__file__) +
                    f'\\Data\\ModelStorage\\W{wait_lable}\\{SIZE}\\{NAME}')
if len(models) == 0:
    print(f'NO MODELS MADE FOR: \\W{wait_lable}\\{SIZE}\\{NAME}')
else:
    for index, mod in enumerate(models):
        modelPath = os.path.dirname(
            __file__)+f'\\Data\\ModelStorage\\W{wait_lable}\\{SIZE}\\{NAME}\\{mod}'
        model = None
        model = load_model(modelPath)

        # Measure accuracy
        pred = model.predict(X)
        pred = np.argmax(pred, axis=1)

        y_true = np.argmax(y, axis=1)

        score = metrics.accuracy_score(y_true, pred)
        if index == 0:
            winner = modelPath
            winnerScore = score
        elif score > winnerScore:
            winner = modelPath
            winnerScore = score

    # %%
    model = None
    model = load_model(winner)

    print("\n\n")
    print('WINNER IS => ' + f'{winner[69:]}')
    print(f'Final time-> {(time.time() - START_TIME)//60} min')
    print()

    model.summary()

    # Measure accuracy
    pred = model.predict(X)
    pred = np.argmax(pred, axis=1)

    y_true = np.argmax(y, axis=1)

    score = metrics.accuracy_score(y_true, pred)
    print("Final accuracy: {}\n".format(score))

    # %%
    # %matplotlib inline

    # Plot a confusion matrix.
    # cm is the confusion matrix, names are the names of the classes.

    def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap='Blues'):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(names))
        plt.xticks(tick_marks, names, rotation=45)
        plt.yticks(tick_marks, names)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    # Compute confusion matrix
    cm = confusion_matrix(y_true, pred)
    print(cm)
    print()
    print(actions)
    print(classification_report(y_true, pred))
    print()
    print("Ploting confusion matrix")
    plt.figure()
    plot_confusion_matrix(cm, actions)
    plt.show()
