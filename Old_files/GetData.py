# %% [markdown]
# Created by: Joseph Crouson
# Version: 0.1
# Date: 12/23/2020
# Purpose: Retrieve the data required

'''
import atexit
import Credentials
import tda
import os
from selenium import webdriver
from tda.client import Client
import pickle
import datetime
import json
import pandas as pd

SYMBOL = r'ES'


def make_webdriver():
    driver = webdriver.Firefox(executable_path=(
        os.path.dirname(__file__)+r'\\Drivers\\geckodriver.exe'))
    atexit.register(lambda: driver.quit())
    return driver


def date_now():
    x = datetime.datetime.today()
    return str(x)


def date_n_days_ago(n=1, string=False):
    x = datetime.datetime.today() - datetime.timedelta(days=n)
    return str(x)


client = tda.auth.easy_client(Credentials.API_KEY, Credentials.REDIRECT_URL,
                              Credentials.TOKEN_PATH, make_webdriver)

resp = client.get_price_history(SYMBOL,
                                start_datetime=datetime.datetime(2020, 1, 20),
                                end_datetime=datetime.datetime(2020, 12, 22),
                                frequency_type=Client.PriceHistory.FrequencyType.MINUTE,
                                frequency=Client.PriceHistory.Frequency.EVERY_MINUTE,
                                need_extended_hours_data=False)
content = json.loads(resp._content.decode())
candleList = content['candles']
df = pd.DataFrame.from_dict(candleList)
# %%
df
# %%
print(candleList[0])
print(type(candleList))
print("Empty? " + str(content['empty']))

dataPickle = open(os.path.dirname(__file__) +
                  f"\\PriceHistory\\{SYMBOL}_10D_1min_data.pickle", "wb")
pickle.dump(resp._content, dataPickle)
dataPickle.close()
'''
