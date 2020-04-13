import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import mplfinance as mpf
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader as web
style.use('ggplot')

df = pd.read_csv('datasets/tsla.csv', index_col=0, parse_dates=True)
#df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()

df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_ohlc['Volume'] = df['Volume'].resample('10D').sum()


for i in df_ohlc.columns:
    df_ohlc.rename(columns={f'{i}':f'{i.capitalize()}'}, inplace=True)


'''df_ohlc['Open'].plot(legend=True)
df_ohlc['High'].plot(legend=True)
df_ohlc['Low'].plot(legend=True)
df_ohlc['Close'].plot(legend=True)'''


print(df_ohlc.corr())


mpf.plot(df_ohlc, type='candle', style='charles', volume=True)
#plt.show()










































