#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop(columns=['Weighted_Price'])
df = df.rename(columns={"Timestamp": "Date"})
df["Date"] = pd.to_datetime(df["Date"], unit='s')
df.set_index('Date', inplace=True)
df['Close']= df['Close'].ffill()
for i in ['High', 'Low', 'Open']:
    df[i].fillna(df['Close'])
df[['Volume_(BTC)', 'Volume_(Currency)']] = (
    df[['Volume_(BTC)', 'Volume_(Currency)']].fillna(0))
df = df.loc[df.index >= "2017-01-01"]
df = df.resample('D').agg({
    'Open': 'mean',
    'High': 'max',
    'Low': 'min',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})
print(df)


df_plot = pd.DataFrame()
df_plot['Open'] = df['Open'].resample('d').mean()
df_plot['High'] = df['High'].resample('d').max()
df_plot['Low'] = df['Low'].resample('d').min()
df_plot['Close'] = df['Close'].resample('d').mean()
df_plot['Volume_(BTC)'] = df['Volume_(BTC)'].resample('d').sum()
df_plot['Volume_(Currency)'] = df['Volume_(Currency)'].resample('d').sum()

df_plot.plot(x_compat=True)

plt.show()