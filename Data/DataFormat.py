'''
Created by: Joseph Crouson
Version: 0.2

Last Modified: 1/2/2021

Purpose:    Given a Symbol, find adjacent folder of that symbol with a folder inside named DataRaw that contains the raw data.
            Clean and preprocess that data.
            Returns a dataframe of processed data.
'''
# %%
# Begin

import pickle
import datetime as dt
import os
import numpy as np
import pandas as pd


#========================================================================================#
#      Process the data for a given symbol and return a dataframe of processed data      #
#                                                                                        #
#                                    Valid inputs                                        #
#                                                                                        #
#         symbol:             Any valid symbol that you have data for                    #
#         aggregationPeriod:  1*, 2, 3, 5                                                #
#         startDate:          String date 'MM/DD/YYYY' or datetime date object           #
#                           ->If not specified then start at the beginning of the data   #
#         futurePredictLen:   Length of time to try and predict                          #
#         priceDelta:         Price Change required for labeling                         #
#                                                                                        #
#========================================================================================#


def process(symbol=None, aggregationPeriod=1, startDate=None, futurePredictLen=None, priceDelta=None):

    if symbol != None:
        try:
            symbol = symbol.upper()
        except Exception as e:
            print(e)
            raise ValueError('symbol error')
    else:
        raise ValueError('symbol not entered\n')

    try:
        aggregationPeriod = int(aggregationPeriod)
        if aggregationPeriod not in [1, 2, 3, 5]:
            raise ValueError('non-valid integer entered\n')
    except Exception as e:
        print(e)
        raise ValueError('valid values [1, 2, 3, 5]\n')

    if isinstance(startDate, (str)):
        try:
            startDate = dt.datetime.strptime(
                startDate, '%m/%d/%Y')
        except Exception as e:
            print(e)
            raise ValueError('invalid startDate, check format\n')
    elif isinstance(startDate, (dt.date, type(None))):
        startDate = dt.datetime(startDate.year, startDate.month, startDate.day)
    else:
        raise ValueError('invalid startDate entry\n')

    if isinstance(futurePredictLen, (str)):
        try:
            futurePredictLen = int(futurePredictLen)
            if futurePredictLen <= 0:
                raise ValueError('futurePredictionLen must be greater than 0')
        except Exception as e:
            print(e)
            raise ValueError('futurePredictionLen entry error')
    elif isinstance(futurePredictLen, (int)):
        if futurePredictLen <= 0:
            raise ValueError('futurePredictionLen must be greater than 0')
        else:
            pass
    else:
        raise ValueError('futurePredictionLen must be a positive int')

    if isinstance(priceDelta, (str)):
        try:
            priceDelta = int(priceDelta)
        except Exception as e:
            print(e)
            raise ValueError('priceDelta entry error')
    elif isinstance(priceDelta, (int)):
        if priceDelta <= 1:
            raise ValueError('priceDelta must be greater than 1')
        else:
            pass
    else:
        raise ValueError('priceDelta must be a positive int')

    # Get a list of data files
    try:
        files = os.listdir(os.path.dirname(__file__) +
                           f'\\SymbolData\\{symbol}\\DataRaw')
    except Exception as e:
        print(e)
        raise ImportError('directory not found')

    # The list will be appended one to the next if there is more than one data file
    # If order matters, it is required that the files be name alphabetically so
    # they append in the proper order
    if len(files) == 1:
        try:
            priceData = pd.read_csv(os.path.dirname(__file__) + f'\\SymbolData\\{symbol}\\DataRaw\\{files[0]}', sep=',', names=[
                                    "DateTime", "Open", "High", "Low", "Close", "Volume"])
        except Exception as e:
            print(e)
            raise ImportError(
                'error with importing data, must be csv\'s, with columns DateTime", "Open", "High", "Low", "Close", "Volume"')
    elif len(files) > 1:
        try:
            priceData = pd.read_csv(os.path.dirname(__file__) + f'\\SymbolData\\{symbol}\\DataRaw\\{files.pop(0)}', sep=',', names=[
                                    "DateTime", "Open", "High", "Low", "Close", "Volume"])
            for index, file in enumerate(files):
                nextFile = pd.read_csv(os.path.dirname(__file__) + f'\\SymbolData\\{symbol}\\DataRaw\\{files[index]}', sep=',', names=[
                                       "DateTime", "Open", "High", "Low", "Close", "Volume"])
                priceData = priceData.append(nextFile, ignore_index=True)
        except Exception as e:
            print(e)
            raise ImportError(
                'error with importing mutltiple data files, must be csv\'s with columns DateTime", "Open", "High", "Low", "Close", "Volume"')
    else:
        raise ImportError('no files found in:\n' +
                          os.path.dirname(__file__)+f'\\SymbolData\\{symbol}\\DataRaw')

    del(files)
    del(file)
    del(nextFile)
    del(index)

    priceData['DateTime'] = pd.to_datetime(
        priceData['DateTime'], format='%Y-%m-%d %H:%M:%S')

    # Check if there is any data after the start date
    if startDate != None:
        if priceData.iloc[priceData.shape[0]-1]['DateTime'] < startDate:
            raise ImportError('no data exists after specified start date')
        else:
            # Strip dates before start
            priceData = priceData[priceData['DateTime'] >= startDate]

    # These 2 lines takes a lot of time, improve here if possible
    priceData['Time'] = priceData['DateTime'].apply(lambda x: x.time())
    priceData['Date'] = priceData['DateTime'].apply(lambda x: x.date())

    # Remove Extended Hours
    priceData = priceData[priceData['Time'] >=
                          dt.time(hour=9, minute=30, second=0)]
    priceData = priceData[priceData['Time'] <=
                          dt.time(hour=15, minute=59, second=0)]

    # Remove holidays
    delDaysSet = {
        dt.date(2010, 1, 1), dt.date(2010, 1, 18), dt.date(2010, 2, 15), dt.date(2010, 4, 2), dt.date(
            2010, 5, 31), dt.date(2010, 7, 5), dt.date(2010, 9, 6), dt.date(2010, 11, 25), dt.date(2010, 12, 24),
        dt.date(2011, 1, 2), dt.date(2011, 1, 3), dt.date(2011, 1, 17), dt.date(2011, 4, 22), dt.date(2011, 5, 30), dt.date(
            2011, 7, 4), dt.date(2011, 9, 5), dt.date(2011, 11, 24), dt.date(2011, 12, 26), dt.date(2011, 2, 21),
        dt.date(2012, 1, 2), dt.date(2012, 1, 16), dt.date(2012, 2, 20), dt.date(2012, 4, 6), dt.date(
            2012, 5, 28), dt.date(2012, 7, 4), dt.date(2012, 9, 3), dt.date(2012, 11, 22), dt.date(2012, 12, 25),
        dt.date(2013, 1, 1), dt.date(2013, 1, 21), dt.date(2013, 2, 18), dt.date(2013, 3, 29), dt.date(
            2013, 5, 27), dt.date(2013, 7, 4), dt.date(2013, 9, 2), dt.date(2013, 11, 28), dt.date(2013, 12, 25),
        dt.date(2014, 1, 1), dt.date(2014, 1, 20), dt.date(2014, 2, 17), dt.date(2014, 4, 18), dt.date(
            2014, 5, 26), dt.date(2014, 7, 4), dt.date(2014, 9, 1), dt.date(2014, 11, 27), dt.date(2014, 12, 25),
        dt.date(2015, 1, 1), dt.date(2015, 1, 19), dt.date(2015, 2, 16), dt.date(2015, 4, 3), dt.date(2015, 5, 25), dt.date(
            2015, 7, 3), dt.date(2015, 9, 7), dt.date(2015, 11, 26), dt.date(2015, 12, 25), dt.date(2015, 7, 6),
        dt.date(2016, 1, 1), dt.date(2016, 1, 18), dt.date(2016, 2, 15), dt.date(2016, 3, 25), dt.date(
            2016, 5, 30), dt.date(2016, 7, 4), dt.date(2016, 9, 5), dt.date(2016, 11, 24), dt.date(2016, 12, 26),
        dt.date(2017, 1, 2), dt.date(2017, 1, 16), dt.date(2017, 2, 20), dt.date(2017, 4, 14), dt.date(2017, 5, 29), dt.date(
            2017, 7, 4), dt.date(2017, 9, 4), dt.date(2017, 11, 23), dt.date(2017, 12, 25), dt.date(2017, 11, 24), dt.date(2017, 11, 27),
        dt.date(2018, 1, 1), dt.date(2018, 1, 15), dt.date(2018, 2, 19), dt.date(2018, 3, 30), dt.date(
            2018, 5, 28), dt.date(2018, 7, 4), dt.date(2018, 9, 3), dt.date(2018, 11, 22), dt.date(2018, 12, 25),
        dt.date(2019, 1, 1), dt.date(2019, 1, 21), dt.date(2019, 2, 18), dt.date(2019, 4, 19), dt.date(
            2019, 5, 27), dt.date(2019, 7, 4), dt.date(2019, 9, 2), dt.date(2019, 11, 28), dt.date(2019, 12, 25),
        dt.date(2020, 1, 1), dt.date(2020, 1, 20), dt.date(2020, 2, 17), dt.date(2020, 4, 10), dt.date(
            2020, 5, 25), dt.date(2020, 7, 3), dt.date(2020, 9, 7), dt.date(2020, 11, 26),
        dt.date(2020, 12, 25), dt.date(2020, 7, 6)}

    priceData['Check'] = ~priceData['Date'].isin(delDaysSet)
    del(delDaysSet)
    priceData = priceData[priceData['Check'] == True]
    priceData.drop(columns=['Check'], inplace=True)

    # Remove Days with missing Candles

    def markDay(row):
        if row['Time2'] != 'NaN' and row['Time'] != dt.time(15, 59, 0):
            if row['Time2'].minute-row['Time'].minute > 1:
                return row['Date']

    priceData['Time2'] = priceData['Time'].shift(-1)
    priceData.iat[-1, len(priceData.columns)-1] = 'NaN'
    # This next line also takes a long time
    priceData['dropDate'] = priceData.apply(markDay, axis=1)
    datesToDrop = priceData['dropDate'].unique()
    priceData = priceData[~priceData['Date'].isin(datesToDrop[1:])]
    del(datesToDrop)
    priceData.drop(columns=['dropDate', 'Time2'], inplace=True)
    priceData.reset_index(drop=True, inplace=True)

    # Aggregate time
    if aggregationPeriod == 1:
        pass
    else:
        priceData['Open'] = priceData['Open'].shift(
            periods=aggregationPeriod-1)
        priceData['DateTime'] = priceData['DateTime'].shift(
            periods=aggregationPeriod-1)
        priceData['Date'] = priceData['Date'].shift(
            periods=aggregationPeriod-1)
        priceData['Time'] = priceData['Time'].shift(
            periods=aggregationPeriod-1)
        priceData.loc[priceData.index[np.arange(
            len(priceData)) % aggregationPeriod == aggregationPeriod-1], 'High'] = priceData.High.rolling(aggregationPeriod).max()
        priceData.loc[priceData.index[np.arange(
            len(priceData)) % aggregationPeriod == aggregationPeriod-1], 'Low'] = priceData.Low.rolling(aggregationPeriod).min()
        priceData.loc[priceData.index[np.arange(
            len(priceData)) % aggregationPeriod == aggregationPeriod-1], 'Volume'] = priceData.Volume.rolling(aggregationPeriod).sum()
        priceData.loc[priceData.index[np.arange(
            len(priceData)) % aggregationPeriod == aggregationPeriod-1], 'Check'] = True
        priceData = priceData[priceData['Check'] == True]
        priceData.drop(columns=['Check'], inplace=True)
        priceData.reset_index(drop=True, inplace=True)

    # Get actions based on futurePredictLen
    priceData = priceData.iloc[::-1]
    priceData.reset_index(drop=True, inplace=True)
    priceData[f'{futurePredictLen}CandleRollingHigh'] = priceData['High'].rolling(
        futurePredictLen).max()
    priceData[f'{futurePredictLen}CandleRollingLow'] = priceData['Low'].rolling(
        futurePredictLen).min()
    priceData = priceData.iloc[::-1]
    priceData.reset_index(drop=True, inplace=True)
    priceData[f'{futurePredictLen}CandleRollingHigh'] = priceData[f'{futurePredictLen}CandleRollingHigh'].shift(
        periods=-1)
    priceData[f'{futurePredictLen}CandleRollingLow'] = priceData[f'{futurePredictLen}CandleRollingLow'].shift(
        periods=-1)

    def getAction(row):
        if row['Close'] + priceDelta < row[f'{futurePredictLen}CandleRollingHigh'] and row['Close'] - priceDelta < row[f'{futurePredictLen}CandleRollingLow']:
            return 'Buy'
        elif row['Close'] + priceDelta > row[f'{futurePredictLen}CandleRollingHigh'] and row['Close'] - priceDelta > row[f'{futurePredictLen}CandleRollingLow']:
            return 'Sell'
        else:
            return 'Wait'

    priceData['Action'] = priceData.apply(getAction, axis=1)
    priceData.drop(columns=[f'{futurePredictLen}CandleRollingHigh',
                            f'{futurePredictLen}CandleRollingLow'], inplace=True)

    # If your are going to add studies to the data, this is where it needs to be done #
    ###################################################################################
    #---------------------------------------------------------------------------------#
    ###################################################################################

    # Normalize (many ways to do this)
    for col in priceData.columns[1:6]:
        priceData[col] = priceData[col].pct_change()

    # Remove rows where the futurePredictLen bleeds into the next day
    priceData['Date2'] = priceData['Date'].shift(periods=-(futurePredictLen-1))
    priceData = priceData[priceData['Date'] == priceData['Date2']]

    # Remove the first candle of each day
    priceData = priceData[priceData['Time'] != dt.time(9, 30, 0)]
    priceData.drop(columns=['Date2'], inplace=True)

    priceData.set_index('DateTime', inplace=True, drop=True)

    for col in priceData.columns[:5]:
        priceData[col] = priceData[col].astype(np.float32)

    # Mark outlier candles
    def markOutlier(row, col, mean, std):
        if row[col] > mean + std*10.25 or row[col] < mean - std*10.25:
            return row['Date']

    for column in priceData.columns[:5]:
        mean = np.mean(priceData[column])
        std = np.std(priceData[column])
        priceData['BadDate'] = priceData.apply(
            markOutlier, args=(col, mean, std), axis=1)

    # Remove days with outlier candles
    datesToDrop = priceData['BadDate'].unique()
    priceData = priceData[~priceData['Date'].isin(datesToDrop[1:])]
    priceData.drop(columns=['BadDate'], inplace=True)

    return priceData


#======================================================================================================#
# %%
if __name__ == '__main__':
    print('''
    #========================================================================================#
    #      Process the data for a given symbol and return a dataframe of processed data      #
    #                                                                                        #
    #                                    Valid inputs                                        #
    #                                                                                        #
    #         symbol:             Any valid symbol that you have data for                    #
    #         aggregationPeriod:  1*, 2, 3, 5                                                #
    #         startDate:          String date 'MM/DD/YYYY' or datetime date object           #
    #                           ->If not specified then start at the beginning of the data   #
    #         futurePredictLen:   Length of time to try and predict                          #
    #         priceDelta:         Price Change required for labeling                         #
    #                                                                                        #
    #========================================================================================#
        ''')
    symbol = input('Enter the symbol -> ')
    if symbol == '':
        symbol = None

    aggregationPeriod = input('Enter the aggregation period -> ')
    if aggregationPeriod == '':
        aggregationPeriod = 1

    startDate = input(
        'Enter the date to start or leave blank to start from the beginning of the data set) \
        \nDates are formatted MM/DD/YYYY -> ')
    if startDate == '':
        startDate = None

    futurePredictLen = input('Enter the Length of time to try and predict -> ')
    if futurePredictLen == '':
        futurePredictLen = None

    priceDelta = input('Price Change required for labeling -> ')
    if priceDelta == '':
        priceDelta = None

    result = process(symbol=symbol, aggregationPeriod=aggregationPeriod,
                     startDate=startDate, futurePredictLen=futurePredictLen, priceDelta=priceDelta)

    del(aggregationPeriod)
    del(futurePredictLen)
    del(priceDelta)
    del(startDate)
    del(symbol)
#======================================================================================================#

# %%
