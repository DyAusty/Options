import pandas as pd
import requests
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import datetime
import numpy as np
import time
import datetime as dt
from ib_insync import *
import yfinance as yf
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from yahoofinancials import YahooFinancials
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels import regression

# Connect to API
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

interval = '1mo' # 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
start = "2018-1-1"
end = datetime.datetime.now()
qty = 1
#symbols = ['AAPL','TSLA','NVDA','META','MSFT','AMZN','GOOG']

# capture last price & beta of each stock:
last_price_AAPL = yf.download('AAPL', start=start, end=end, interval=interval)
last_price_AAPL = last_price_AAPL['Close'].values[-1:]
last_price_AAPL = round(float(last_price_AAPL),2)
print("Last Price AAPL:")
print(last_price_AAPL)

symbols_AAPL = ['AAPL', 'SPY']
data_AAPL = yf.download(symbols_AAPL, start)['Adj Close']
price_change_AAPL = data_AAPL.pct_change()
df_AAPL = price_change_AAPL.drop(price_change_AAPL.index[0])
x_AAPL = np.array(df_AAPL['SPY']).reshape((-1,1))
y_AAPL = np.array(df_AAPL['AAPL'])
model_AAPL = LinearRegression().fit(x_AAPL, y_AAPL)
beta_AAPL = model_AAPL.coef_
beta_AAPL = round(float(beta_AAPL),4)
print("BETA AAPL:")
print(beta_AAPL)
beta_AAPL_adjusted_notional = last_price_AAPL * beta_AAPL * qty
print("BETA AAPL ADJUSTED NOTIONAL:")
print(beta_AAPL_adjusted_notional)


last_price_TSLA = yf.download('TSLA', start=start, end=end, interval=interval)
last_price_TSLA = last_price_TSLA['Close'].values[-1:]
last_price_TSLA = round(float(last_price_TSLA),2)
print("Last Price TSLA:")
print(last_price_TSLA)
symbols_TSLA = ['TSLA', 'SPY']
data_TSLA = yf.download(symbols_TSLA, start)['Adj Close']
price_change_TSLA = data_TSLA.pct_change()
df_TSLA = price_change_TSLA.drop(price_change_TSLA.index[0])
x_TSLA = np.array(df_TSLA['TSLA']).reshape((-1,1))
y_TSLA = np.array(df_TSLA['SPY'])
model_TSLA = LinearRegression().fit(x_TSLA, y_TSLA)
beta_TSLA = model_TSLA.coef_
beta_TSLA = round(float(beta_TSLA),4)
print("BETA TSLA:")
print(beta_TSLA)
beta_TSLA_adjusted_notional = last_price_TSLA * beta_TSLA * qty
print("BETA TSLA ADJUSTED NOTIONAL:")
print(beta_TSLA_adjusted_notional)



last_price_NVDA = yf.download('NVDA', start=start, end=end, interval=interval)
last_price_NVDA = last_price_NVDA['Close'].values[-1:]
last_price_NVDA = round(float(last_price_NVDA),2)
print("Last Price NVDA:")
print(last_price_NVDA)
symbols_NVDA = ['NVDA', 'SPY']
data_NVDA = yf.download(symbols_NVDA, start)['Adj Close']
price_change_NVDA = data_NVDA.pct_change()
df_NVDA = price_change_NVDA.drop(price_change_NVDA.index[0])
x_NVDA = np.array(df_NVDA['SPY']).reshape((-1,1))
y_NVDA = np.array(df_NVDA['NVDA'])
model_NVDA = LinearRegression().fit(x_NVDA, y_NVDA)
beta_NVDA = model_NVDA.coef_
beta_NVDA = round(float(beta_NVDA),4)
print("BETA NVDA:")
print(beta_NVDA)
beta_NVDA_adjusted_notional = last_price_NVDA * beta_NVDA * qty
print("BETA NVDA ADJUSTED NOTIONAL:")
print(beta_NVDA_adjusted_notional)


last_price_GOOG = yf.download('GOOG', start=start, end=end, interval=interval)
last_price_GOOG = last_price_GOOG['Close'].values[-1:]
last_price_GOOG = round(float(last_price_GOOG),2)
print("Last Price GOOG:")
print(last_price_GOOG)
symbols_GOOG = ['GOOG', 'SPY']
data_GOOG = yf.download(symbols_GOOG, start)['Adj Close']
price_change_GOOG = data_GOOG.pct_change()
df_GOOG = price_change_GOOG.drop(price_change_GOOG.index[0])
x_GOOG = np.array(df_GOOG['SPY']).reshape((-1,1))
y_GOOG = np.array(df_GOOG['GOOG'])
model_GOOG = LinearRegression().fit(x_GOOG, y_GOOG)
beta_GOOG = model_GOOG.coef_
beta_GOOG = round(float(beta_GOOG),4)
print("BETA GOOG:")
print(beta_GOOG)
beta_GOOG_adjusted_notional = last_price_GOOG * beta_GOOG * qty
print("BETA GOOG ADJUSTED NOTIONAL:")
print(beta_GOOG_adjusted_notional)


last_price_META = yf.download('META', start=start, end=end, interval=interval)
last_price_META = last_price_META['Close'].values[-1:]
last_price_META = round(float(last_price_META),2)
print("Last Price META:")
print(last_price_META)
symbols_META = ['META', 'SPY']
data_META = yf.download(symbols_META, start)['Adj Close']
price_change_META = data_META.pct_change()
df_META = price_change_META.drop(price_change_META.index[0])
x_META = np.array(df_META['SPY']).reshape((-1,1))
y_META = np.array(df_META['META'])
model_META = LinearRegression().fit(x_META, y_META)
beta_META = model_META.coef_
beta_META = round(float(beta_META),4)
print("BETA META:")
print(beta_META)
beta_META_adjusted_notional = last_price_META * beta_META * qty
print("BETA META ADJUSTED NOTIONAL:")
print(beta_META_adjusted_notional)


last_price_MSFT = yf.download('MSFT', start=start, end=end, interval=interval)
last_price_MSFT = last_price_MSFT['Close'].values[-1:]
last_price_MSFT = round(float(last_price_MSFT),2)
print("Last Price MSFT:")
print(last_price_MSFT)
symbols_MSFT = ['MSFT', 'SPY']
data_MSFT = yf.download(symbols_MSFT, start)['Adj Close']
price_change_MSFT = data_MSFT.pct_change()
df_MSFT = price_change_MSFT.drop(price_change_MSFT.index[0])
x_MSFT = np.array(df_MSFT['SPY']).reshape((-1,1))
y_MSFT = np.array(df_MSFT['MSFT'])
model_MSFT = LinearRegression().fit(x_MSFT, y_MSFT)
beta_MSFT = model_MSFT.coef_
beta_MSFT = round(float(beta_MSFT),4)
print("BETA MSFT:")
print(beta_MSFT)
beta_MSFT_adjusted_notional = last_price_MSFT * beta_MSFT * qty
print("BETA MSFT ADJUSTED NOTIONAL:")
print(beta_MSFT_adjusted_notional)


last_price_AMZN = yf.download('AMZN', start=start, end=end, interval=interval)
last_price_AMZN = last_price_AMZN['Close'].values[-1:]
last_price_AMZN = round(float(last_price_AMZN),2)
print("Last Price AMZN:")
print(last_price_AMZN)
symbols_AMZN = ['AMZN', 'SPY']
data_AMZN = yf.download(symbols_AMZN, start)['Adj Close']
price_change_AMZN = data_AMZN.pct_change()
df_AMZN = price_change_AMZN.drop(price_change_AMZN.index[0])
x_AMZN = np.array(df_AMZN['SPY']).reshape((-1,1))
y_AMZN = np.array(df_AMZN['AMZN'])
model_AMZN = LinearRegression().fit(x_AMZN, y_AMZN)
beta_AMZN = model_AMZN.coef_
beta_AMZN = round(float(beta_AMZN),4)
print("BETA AMZN:")
print(beta_AMZN)
beta_AMZN_adjusted_notional = last_price_AMZN * beta_AMZN * qty
print("BETA AMZN ADJUSTED NOTIONAL:")
print(beta_AMZN_adjusted_notional)

####################################################################################################################################
####################################################################################################################################
# Extract Futures Last Price:
entry_df_futures = yf.download('ES=F', period="1d", interval="1m")
last_price_futures = round(float(entry_df_futures['Close'].values[-1:]), 2)
print("Last Price Futures:")
print(last_price_futures)
futures_notional = last_price_futures * 50
print("Futures Notional Value:")
print(futures_notional)

total_beta_adjusted_notional_value_stocks = beta_AAPL_adjusted_notional + beta_TSLA_adjusted_notional \
                                            + beta_META_adjusted_notional + beta_GOOG_adjusted_notional \
                                            + beta_AMZN_adjusted_notional + beta_NVDA_adjusted_notional \
                                            + beta_MSFT_adjusted_notional
print("Total Beta Adjusted Notional Value All Stocks:")
print(total_beta_adjusted_notional_value_stocks)

# Effective beta hedge requires that notional of the hedging trade is equivalent
# to the beta-adjusted notional value of single stock:

# Spread Trade Requires:

# Take into account 25 shares of each of the 7 names for the portfolio on a given day:
num_shares_ticker = 25
beta_adjusted_notional_value_entire_portfolio = total_beta_adjusted_notional_value_stocks * num_shares_ticker
print("Beta Adjusted Notional Value Entire Portfolio:")
print(beta_adjusted_notional_value_entire_portfolio)

# Calculate hedge on how many contracts to short on $ES:
hedged_futures_amount = beta_adjusted_notional_value_entire_portfolio / last_price_futures
print("Number of Futures Contracts to Hedge Portfolio with:")
hedged_futures_amount = int(hedged_futures_amount)
print(hedged_futures_amount)
