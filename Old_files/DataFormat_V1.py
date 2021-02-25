# %% [markdown]
# Created by: Joseph Crouson
# Version: 0.1
# Date: 12/23/2020
# Last Modified: 12/26/2020
# Purpose: Clean and preprocess the data

# %%
import pandas as pd
import numpy as np
import os
import datetime
import pickle

# %%
# Get data from files
priceData_2010_thru_2019 = pd.read_csv(os.path.dirname(__file__) +
                                       r"\PriceHistory\NQ_2010_2019_continuous.txt", sep=",", names=["DateTime", "Open", "High", "Low", "Close", "Volume"])

priceData_2020_to_present = pd.read_csv(os.path.dirname(__file__) +
                                        r"\PriceHistory\NQ_2020_2020_continuous.txt", sep=",", names=["DateTime", "Open", "High", "Low", "Close", "Volume"])

# %%
time1 = priceData_2010_thru_2019['DateTime']
time2 = priceData_2020_to_present['DateTime']
time1 = pd.to_datetime(time1, format='%Y-%m-%d %H:%M:%S')
time2 = pd.to_datetime(time2, format='%Y-%m-%d %H:%M:%S')
time1 = time1.apply(lambda x: x.time())
time2 = time2.apply(lambda x: x.time())
priceData_2010_thru_2019['Time'] = time1
priceData_2020_to_present['Time'] = time2

# %%
priceData = priceData_2010_thru_2019.append(
    priceData_2020_to_present, ignore_index=True)

# %%
priceData = priceData[priceData['Time'] >=
                      datetime.time(hour=9, minute=30, second=0)]
priceData = priceData[priceData['Time'] <=
                      datetime.time(hour=16, minute=1, second=0)]

priceData.reset_index(drop=True, inplace=True)

del(priceData_2010_thru_2019)
del(priceData_2020_to_present)
del(time1)
del(time2)

# %%
# There are market holidays that need to be removed
datetime_series = pd.to_datetime(priceData['DateTime'])
priceData['DateTime'] = datetime_series
del(datetime_series)
delDaysSet = {datetime.date(2010, 1, 1), datetime.date(2010, 1, 18), datetime.date(2010, 2, 15), datetime.date(2010, 4, 2), datetime.date(2010, 5, 31), datetime.date(2010, 7, 5), datetime.date(2010, 9, 6), datetime.date(2010, 11, 25), datetime.date(2010, 12, 24),
              datetime.date(2011, 1, 2), datetime.date(2011, 1, 3), datetime.date(2011, 1, 17), datetime.date(2011, 4, 22), datetime.date(2011, 5, 30), datetime.date(
    2011, 7, 4), datetime.date(2011, 9, 5), datetime.date(2011, 11, 24), datetime.date(2011, 12, 26), datetime.date(2011, 2, 21),
    datetime.date(2012, 1, 2), datetime.date(2012, 1, 16), datetime.date(2012, 2, 20), datetime.date(2012, 4, 6), datetime.date(
    2012, 5, 28), datetime.date(2012, 7, 4), datetime.date(2012, 9, 3), datetime.date(2012, 11, 22), datetime.date(2012, 12, 25),
    datetime.date(2013, 1, 1), datetime.date(2013, 1, 21), datetime.date(2013, 2, 18), datetime.date(2013, 3, 29), datetime.date(
    2013, 5, 27), datetime.date(2013, 7, 4), datetime.date(2013, 9, 2), datetime.date(2013, 11, 28), datetime.date(2013, 12, 25),
    datetime.date(2014, 1, 1), datetime.date(2014, 1, 20), datetime.date(2014, 2, 17), datetime.date(2014, 4, 18), datetime.date(
    2014, 5, 26), datetime.date(2014, 7, 4), datetime.date(2014, 9, 1), datetime.date(2014, 11, 27), datetime.date(2014, 12, 25),
    datetime.date(2015, 1, 1), datetime.date(2015, 1, 19), datetime.date(2015, 2, 16), datetime.date(2015, 4, 3), datetime.date(2015, 5, 25), datetime.date(
    2015, 7, 3), datetime.date(2015, 9, 7), datetime.date(2015, 11, 26), datetime.date(2015, 12, 25), datetime.date(2015, 7, 6),
    datetime.date(2016, 1, 1), datetime.date(2016, 1, 18), datetime.date(2016, 2, 15), datetime.date(2016, 3, 25), datetime.date(
    2016, 5, 30), datetime.date(2016, 7, 4), datetime.date(2016, 9, 5), datetime.date(2016, 11, 24), datetime.date(2016, 12, 26),
    datetime.date(2017, 1, 2), datetime.date(2017, 1, 16), datetime.date(2017, 2, 20), datetime.date(2017, 4, 14), datetime.date(2017, 5, 29), datetime.date(
    2017, 7, 4), datetime.date(2017, 9, 4), datetime.date(2017, 11, 23), datetime.date(2017, 12, 25), datetime.date(2017, 11, 24), datetime.date(2017, 11, 27),
    datetime.date(2018, 1, 1), datetime.date(2018, 1, 15), datetime.date(2018, 2, 19), datetime.date(2018, 3, 30), datetime.date(
    2018, 5, 28), datetime.date(2018, 7, 4), datetime.date(2018, 9, 3), datetime.date(2018, 11, 22), datetime.date(2018, 12, 25),
    datetime.date(2019, 1, 1), datetime.date(2019, 1, 21), datetime.date(2019, 2, 18), datetime.date(2019, 4, 19), datetime.date(
    2019, 5, 27), datetime.date(2019, 7, 4), datetime.date(2019, 9, 2), datetime.date(2019, 11, 28), datetime.date(2019, 12, 25),
    datetime.date(2020, 1, 1), datetime.date(2020, 1, 20), datetime.date(2020, 2, 17), datetime.date(2020, 4, 10), datetime.date(2020, 5, 25), datetime.date(2020, 7, 3), datetime.date(2020, 9, 7), datetime.date(2020, 11, 26), datetime.date(2020, 12, 25), datetime.date(2020, 7, 6)}
priceData['Date'] = priceData['DateTime'].apply(lambda x: x.date())
priceData['Check'] = ~priceData['Date'].isin(delDaysSet)
priceData = priceData[priceData['Check'] == True]
priceData.reset_index(drop=True, inplace=True)
priceData.drop(columns=['Date', 'Check'], inplace=True)
del(delDaysSet)
# %%
# Remove Days with missing Candles


def missingCandleSeries(s):
    dropList = []
    dropList.append(False)
    for index, value in enumerate(s[1:], start=1):
        if value.minute-1 != s[index-1].minute:
            if value == datetime.time(9, 30) or (value.minute == 0 and s[index-1].minute == 59):
                dropList.append(False)
            else:
                dropList.append(True)
        else:
            dropList.append(False)
    dropSeries = pd.Series(data=dropList)
    return dropSeries


def getDateToDrop(row):
    if row['Check']:
        return row['DateTime'].date()
    else:
        return None


def removeShortDays(df, s):
    dropSet = set()
    for date in s:
        temp = df[df['Date'] == date]
        if temp.shape[0] == 392:
            pass
        else:
            print(date)
            dropSet.add(date)
    return dropSet


priceData['Check'] = missingCandleSeries(priceData['Time'])
priceData['Date'] = priceData.apply(getDateToDrop, axis=1)
datesToDrop = priceData['Date'].unique()
priceData = priceData[~priceData['DateTime'].apply(
    lambda x: x.date()).isin(datesToDrop[1:])]
del(datesToDrop)
priceData.reset_index(drop=True, inplace=True)
priceData.drop(columns=['Time', 'Check', 'Date'], inplace=True)
priceData['Date'] = priceData['DateTime'].apply(lambda x: x.date())
uniqueDates = priceData['Date'].unique()
datesToDrop = removeShortDays(priceData, uniqueDates)
del(uniqueDates)
priceData = priceData[~priceData['Date'].isin(datesToDrop)]
del(datesToDrop)
priceData.reset_index(drop=True, inplace=True)
# %%
# Make records 2 minutes
priceData.rename(columns={'High': 'Start_High',
                          'Low': 'Start_Low', 'Volume': 'Start_Volume'}, inplace=True)
priceData['End_High'] = priceData['Start_High']
priceData['End_High'] = priceData['End_High'].shift(periods=-1)
priceData['End_Low'] = priceData['Start_Low']
priceData['End_Low'] = priceData['End_Low'].shift(periods=-1)
priceData['End_Volume'] = priceData['Start_Volume']
priceData['End_Volume'] = priceData['End_Volume'].shift(periods=-1)


def getHigh(row):
    return max(row['Start_High'], row['End_High'])


def getLow(row):
    return min(row['Start_Low'], row['End_Low'])


def getVolume(row):
    return row['Start_Volume'] + row['End_Volume']


priceData['High'] = priceData.apply(getHigh, axis=1)
priceData['Low'] = priceData.apply(getLow, axis=1)
priceData['Volume'] = priceData.apply(getVolume, axis=1)
priceData['Close'] = priceData['Close'].shift(periods=-1)

priceData.drop(columns=['Start_High', 'Start_Low',
                        'Start_Volume', 'End_High', 'End_Low', 'End_Volume'], inplace=True)
priceData = priceData[['DateTime', 'Open',
                       'High', 'Low', 'Close', 'Volume']]
priceData = priceData[priceData.index % 2 == 0]
priceData = priceData[priceData['DateTime'].apply(
    lambda x: x.time()) != datetime.time(16, 0, 0)]
priceData.reset_index(drop=True, inplace=True)

print('Data Is Now In 2 min Bars')

# %%
# Save the Data
dataPickle = open(os.path.dirname(__file__) +
                  r"\PriceHistory\NQ_Data.pickle", "wb")
pickle.dump(priceData, dataPickle)
dataPickle.close()
del(dataPickle)
del(priceData)

# %%
# Load the Data
pickle_in = open(os.path.dirname(__file__) +
                 r"\PriceHistory\NQ_Data.pickle", 'rb')
priceData = pickle.load(pickle_in)
pickle_in.close()
del(pickle_in)

# %%
# Build the action column

day = 0
frontPadding = ['Na', 'Na', 'Na', 'Na', 'Na', 'Na', 'Na', 'Na', 'Na', 'Na']
actionList = []
backPadding = ['Na', 'Na']


def thisDaysActions(dailyData):
    global day
    global actionList
    global frontPadding
    global backPadding
    actionList.extend(frontPadding)
    for thisBar in range(10, 193):
        subFrame = dailyData.iloc[thisBar:thisBar+3]
        currentPrice = subFrame['Open'].iloc[0]
        highestFutureHigh = subFrame['High'].max()
        lowestFutureLow = subFrame['Low'].min()
        action = 'Wait'
        if highestFutureHigh >= currentPrice + 7.5:
            action = 'Buy'
        elif lowestFutureLow <= currentPrice - 7.5:
            action = 'Sell'
        elif (lowestFutureLow > currentPrice - 7.5) and (highestFutureHigh < currentPrice + 7.5):
            action = 'Wait'
        else:
            action = 'Check'
        actionList.append(action)
    actionList.extend(backPadding)
    print(f'Day {day} completed')
    day += 1


def getActions(uniqueDates):
    global priceData
    for date in uniqueDates:
        temp = priceData[priceData['Date'] == date]
        thisDaysActions(temp)
    return pd.Series(actionList)


uniqueDates = priceData['DateTime'].apply(lambda x: x.date()).unique()
priceData['Date'] = priceData['DateTime'].apply(lambda x: x.date())
getActions(uniqueDates)
priceData['Action'] = getActions(uniqueDates)
del(day)
del(actionList)
del(frontPadding)
del(backPadding)
del(uniqueDates)
priceData.drop(columns=['Date'], inplace=True)
print('Done')

# %%
# Save data with action
dataPickle = open(os.path.dirname(__file__) +
                  r"\PriceHistory\NQ_Data_With_Action.pickle", "wb")
pickle.dump(priceData, dataPickle)
dataPickle.close()
del(dataPickle)
del(priceData)

# %%
# Load the Data with actions
pickle_in = open(os.path.dirname(__file__) +
                 r"\PriceHistory\NQ_Data_With_Action.pickle", 'rb')
priceData = pickle.load(pickle_in)
pickle_in.close()
del(pickle_in)
# If your are going to add studies to the data, this is where it needs to be done

# From here on I am using dates from June 2015 and later
priceData = priceData[priceData['DateTime']
                      > datetime.datetime(2015, 6, 1, 0, 0, 0)]
priceData.reset_index(drop=True, inplace=True)


# If your are going to add studies to the data, this is where it needs to be done #
###################################################################################
#---------------------------------------------------------------------------------#
###################################################################################

# %%
# Data normalization
countUniqueDates = priceData['DateTime'].apply(
    lambda x: x.date()).unique().size
priceData['PrevClose'] = priceData['Close'].shift(1)

for i in range(countUniqueDates):
    priceData.iat[i*195, 7] = priceData.iloc[i*195]['Open']
print('Shift Complete')


def normalizeOpen(row):
    return (row['Open'] - row['PrevClose']) / row['PrevClose']


def normalizeHigh(row):
    return (row['High'] - row['PrevClose']) / row['PrevClose']


def normalizeLow(row):
    return (row['Low'] - row['PrevClose']) / row['PrevClose']


def normalizeClose(row):
    return (row['Close'] - row['PrevClose']) / row['PrevClose']


priceData['Open'] = priceData.apply(normalizeOpen, axis=1)
priceData['High'] = priceData.apply(normalizeHigh, axis=1)
priceData['Low'] = priceData.apply(normalizeLow, axis=1)
priceData['Close'] = priceData.apply(normalizeClose, axis=1)
print('Normalization Complete')

priceData.Open = priceData.Open.astype(np.float32)
priceData.High = priceData.High.astype(np.float32)
priceData.Low = priceData.Low.astype(np.float32)
priceData.Close = priceData.Close.astype(np.float32)
print('Type Change Complete')

priceData.drop(columns=['PrevClose'], inplace=True)
print('Done')

# %%
# Create Records
priceData['Action'] = priceData['Action'].shift(periods=-1)
indexesToDropList = []
for i in range(countUniqueDates):
    indexesToDropList.extend([i*195+192, i*195+193, i*195+194])
priceData.drop(index=indexesToDropList, inplace=True)
priceData.reset_index(drop=True, inplace=True)
del(indexesToDropList)
del(countUniqueDates)

# If you want to include Volume add it to the list, else drop volume
priceData.drop(columns=['Volume', 'DateTime'], inplace=True)
labels = ['Open', 'High', 'Low', 'Close']
for i in range(1, 10):
    for label in labels:
        priceData[f'{label}_{i}_CandleAgo'] = priceData[label].shift(periods=i)

del(i)
del(label)
del(labels)
priceData = priceData[priceData['Action'] != 'Na']
priceData.reset_index(drop=True, inplace=True)

actions = priceData['Action']
priceData.drop(columns=['Action'], inplace=True)
priceData['Action'] = actions
del(actions)

# %%
# Save the finalized data
dataPickle = open(os.path.dirname(__file__) +
                  r"\PriceHistory\NQ_Data_ProperRecords_Normalized.pickle", "wb")
pickle.dump(priceData, dataPickle)
dataPickle.close()
del(dataPickle)
del(priceData)

# %%
# Just a test to make sure it's all there still
# pickle_in = open(os.path.dirname(__file__) +
#                 r"\PriceHistory\NQ_Data_ProperRecords_Normalized.pickle", 'rb')
#priceData = pickle.load(pickle_in)
# pickle_in.close()
# del(pickle_in)

# %%
