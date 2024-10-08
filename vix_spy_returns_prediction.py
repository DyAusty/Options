import yfinance as yf
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

start_date = '2000-01-01'
end_date = '2023-09-23'
spy_data = yf.download('SPY', start=start_date, end=end_date)['Close']
vix_data = yf.download('^VIX', start=start_date, end=end_date)['Close']

spy_returns = spy_data.pct_change().dropna()
vix_returns = vix_data.pct_change().dropna()

data = pd.DataFrame({'SPY_returns': spy_returns, 'VIX_returns': vix_returns})

r_squared_values = []
p_values = []
forward_return_periods = list(range(1, 31))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 9))

for period in forward_return_periods:
    data[f'SPY_{period}d_return'] = data['SPY_returns'].shift(-period)

    data.dropna(inplace=True)

    X = sm.add_constant(data['VIX_returns'])
    model = sm.OLS(data[f'SPY_{period}d_return'], X).fit()

    r_squared_values.append(model.rsquared)
    p_values.append(model.pvalues['VIX_returns'])

ax1.plot(forward_return_periods, r_squared_values, marker='o', linestyle='-', label='R-squared')
ax1.set_title('R-squared vs. SPY Forward Return Period')
ax1.set_xlabel('SPY Forward Return Period (Days)')
ax1.set_ylabel('R-squared Value')
ax1.legend()
ax1.grid(True)
ax2.plot(forward_return_periods, p_values, marker='x', linestyle='--', color='red', label='P-value')
ax2.set_title('P-value vs. SPY Forward Return Period')
ax2.set_xlabel('SPY Forward Return Period (Days)')
ax2.set_ylabel('P-value')
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.show()
