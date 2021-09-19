# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 14:32:48 2021

@author: M Shoaib
"""



#this will download stock to csv
# ticker = 'AAPL'
# period1 = int(time.mktime(datetime.datetime(2020, 12, 1, 23, 59).timetuple()))
# period2 = int(time.mktime(datetime.datetime(2020, 12, 31, 23, 59).timetuple()))
# interval = '1d' # 1d, 1m

# query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'

# df = pd.read_csv(query_string)
# # print(df)
# df.to_csv('AAPL.csv')
##################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM,Dropout,Activation
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
stock_price = pd.read_csv('AAPL.csv')
headlines = pd.read_csv('india-news-headlines.csv')
stock_price.shape
stock_price.dropna(axis=0,inplace=True)
stock_price.dropna(axis=0,inplace=True)
stock_price['Date'] = pd.to_datetime(stock_price['Date']).dt.normalize()
stock_price.sort_values('Date',ascending=True)
stock_price.set_index('Date',inplace=True)
stock_price.head(5)
headlines.duplicated().sum()
headlines = headlines.drop_duplicates()
headlines.head(5)
headlines.isnull().sum()
headlines = headlines.filter(['publish_date','headline_text'])
headlines['publish_date'] = headlines['publish_date'].astype(str)
headlines['publish_date'] = headlines['publish_date'].apply(lambda x : x[0:4]+'-'+x[4:6]+'-'+x[6:])
headlines['publish_date'] = pd.to_datetime(headlines['publish_date']).dt.normalize()
headlines = headlines.groupby('publish_date')['headline_text'].apply(lambda x : ''.join(x)).reset_index()
headlines.set_index('publish_date',inplace=True)
headlines.sort_index()

df = pd.concat([stock_price,headlines],axis=1)
df.dropna(axis=0,inplace=True)
df.head(10)
sid = SentimentIntensityAnalyzer()

df['compound'] = df['headline_text'].apply(lambda x : sid.polarity_scores(x)['compound'])
df.head(5)
df.isnull().sum()
df.describe()
df.info()
df['Close'].plot()
plt.title('Close Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()
df.rolling(7).mean().head(20)

df['Close'].plot()
df.rolling(window=30).mean()['Close'].plot()
plt.title('Close Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()
use_data_size = len(df) - 1
train_data_size = int(0.8 * use_data_size)
total_data_size = len(df)
start_point = total_data_size - use_data_size
print("Length of Training Set is ",train_data_size)
print('Length of Testing Set is ', total_data_size - train_data_size)
close_price = df.iloc[start_point:total_data_size,3]
compound = df['compound']
open_price = df.iloc[start_point:total_data_size,0]
high = df.iloc[start_point:total_data_size,1]
low = df.iloc[start_point:total_data_size,2]
volume = df.iloc[start_point:total_data_size,5]
close_price_shifted = close_price.shift(-1)
compound_shifted = compound.shift(-1)
stock_price_data = pd.DataFrame({
    'close':close_price,
    'close_price_shifted':close_price_shifted,
    'compound':df['compound'],
    'compound_shifted':compound_shifted,
    'open':open_price,
    'high':high,
    'low':low,
    'volume':volume
                                })
stock_price_data.head()
stock_price_data.dropna(axis=0,inplace=True)

y = stock_price_data['close_price_shifted']
y.shape

x = stock_price_data.drop(['close_price_shifted'],axis=1)
x.head()
scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
y = scaler.fit_transform(np.array(y).reshape(-1,1))
x = scaler.fit_transform(x)
X_train = x[:train_data_size,]
X_test = x[train_data_size + 1: len(x),]
y_train = y[:train_data_size,]
y_test = y[train_data_size + 1 : len(y),]
print('Size of Training set X : ',X_train.shape)
print('Size of Test set X : ',X_test.shape)
print('Size of Training set Y : ',y_train.shape)
print('Size of Testing set Y : ',y_test.shape)
X_train = X_train.reshape(-1,7,1)
X_test = X_test.reshape(-1,7,1)
model = Sequential()
model.add(LSTM(100, return_sequences=True, activation='tanh',input_shape= X_train.shape[1:]))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True, activation='tanh'))
model.add(Dropout(0.2))
model.add(LSTM(100,return_sequences=False,activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1))

model.summary()
model.compile(optimizer='rmsprop',loss='mse')
history = model.fit(X_train,y_train, epochs=20, batch_size=10, validation_data=(X_test,y_test))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
model_json = model.to_json()
with open('model.json','w') as file :
    file.write(model_json)
model.save_weights('model.h5')