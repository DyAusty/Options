import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import norm
import matplotlib.pyplot as plt
from copulae import GaussianCopula
import warnings
warnings.filterwarnings('ignore')
# Define ticker of interest to compare if $SPX draws down to:
#############################################################
############################################################
#USER INPUTS (3 INPUTS):
portfolio_tickers = ['NVDA']
market_drop_percentage = -0.03
end_date = "2024-05-08"
#############################################################
#####################################################################################################################################
start_date = "2019-05-08"
symbols = ["^GSPC"] + portfolio_tickers
data = yf.download(symbols, start=start_date, end=end_date)["Adj Close"]
#yf.download('^GSPC', period="5d", interval="5m")
#yf.download(symbols, start=start_date, end=end_date)["Adj Close"]
# 2. Compute daily returns
returns = data.pct_change().dropna()
# 3. Transform returns to [0,1] range using quantile transformation
quantile_transformers = {}
data_uniform = pd.DataFrame()
for symbol in returns.columns:
    qt = QuantileTransformer(output_distribution='uniform')
    data_uniform[symbol] = qt.fit_transform(returns[[symbol]]).flatten()
    quantile_transformers[symbol] = qt
# Define the function for conditional sampling
def conditional_sample(u1, rho, n_samples=1000):
    u2 = np.linspace(0.001, 0.999, n_samples)
    return u2, norm.cdf((norm.ppf(u1) - rho * norm.ppf(u2)) / np.sqrt(1 - rho ** 2))
# Simulation for each stock in the portfolio
results = {}
for ticker in portfolio_tickers:
    index_drop = quantile_transformers["^GSPC"].transform(np.array([[market_drop_percentage]]))[0][0]
    # Bivariate assumption with ^GSPC and each stock
    bi_data = data_uniform[["^GSPC", ticker]]
    bi_copula = GaussianCopula(dim=2)
    bi_copula.fit(bi_data.values)
    rho = bi_copula.params[0]
    conditional_u2, conditional_cdf = conditional_sample(index_drop, rho)
    conditional_returns = quantile_transformers[ticker].inverse_transform(conditional_cdf.reshape(-1, 1)).flatten()
    results[ticker] = conditional_returns

# Compute portfolio returns from individual stock returns
portfolio_returns = np.mean(np.array([results[ticker] for ticker in portfolio_tickers]), axis=0)

# Visualization for the Portfolio
fig, ax = plt.subplots(2, 2, figsize=(12, 7))
last_known_prices = data[portfolio_tickers].iloc[-1]
portfolio_last_known_price = np.mean(last_known_prices)  # Equally weighted

min_return = np.min(portfolio_returns)
max_return = np.max(portfolio_returns)
mean_return = np.mean(portfolio_returns)

final_min_price = portfolio_last_known_price * (1 + min_return)
final_max_price = portfolio_last_known_price * (1 + max_return)
final_mean_price = portfolio_last_known_price * (1 + mean_return)

simulated_dates = pd.date_range(start=data.index[-1], periods=31, freq='D')[1:]

min_price_trajectory = [portfolio_last_known_price] + [final_min_price] * (len(simulated_dates) - 1)
max_price_trajectory = [portfolio_last_known_price] + [final_max_price] * (len(simulated_dates) - 1)
mean_price_trajectory = [portfolio_last_known_price] + [final_mean_price] * (len(simulated_dates) - 1)

# Plot 1: Histogram of Simulated Returns
ax[0, 0].hist(portfolio_returns, bins=50, edgecolor='k', alpha=0.7)
ax[0, 0].set_title(f"Simulated Portfolio Returns given {market_drop_percentage * 100:.2f}% drop in S&P 500",
                   fontsize=11)
ax[0, 0].set_xlabel("Returns")
ax[0, 0].set_ylabel("Frequency")

# Plot 2: CDF of Simulated Returns
ax[0, 1].hist(portfolio_returns, bins=100, density=True, cumulative=True, alpha=0.7)
ax[0, 1].set_title('CDF of Simulated Portfolio Returns', fontsize=11)

# Plot 3: KDE of Simulated Returns
sns.kdeplot(portfolio_returns, shade=True, ax=ax[1, 0])
ax[1, 0].set_title('KDE of Simulated Portfolio Returns', fontsize=11)

# Plot 4: Portfolio Original vs. Worst, Best, and Mean Case Scenarios
portfolio_prices = data[portfolio_tickers].mean(axis=1)
portfolio_prices.plot(ax=ax[1, 1], label="Original Prices")
pd.Series(min_price_trajectory, index=simulated_dates).plot(ax=ax[1, 1], label="Worst-Case Scenario", linestyle='--',
                                                            color="red")
pd.Series(max_price_trajectory, index=simulated_dates).plot(ax=ax[1, 1], label="Best-Case Scenario", linestyle='--',
                                                            color="green")
pd.Series(mean_price_trajectory, index=simulated_dates).plot(ax=ax[1, 1], label="Mean Scenario", linestyle='--',
                                                             color="blue")
label_x_position = simulated_dates[-10]
ax[1, 1].annotate(f"{min_return * 100:.2f}% (Worst Scenario)", (label_x_position, final_min_price * 0.98), fontsize=7,
                  ha="left", color="red")
ax[1, 1].annotate(f"{max_return * 100:.2f}% (Best Scenario)", (label_x_position, final_max_price * 1.02), fontsize=7,
                  ha="left", color="green")
ax[1, 1].annotate(f"{mean_return * 100:.2f}% (Mean Scenario)", (label_x_position, final_mean_price), fontsize=7,
                  ha="left", color="blue")
ax[1, 1].set_title(f"Portfolio Original vs. Worst, Best, and Mean Case Scenarios", fontsize=11)
ax[1, 1].legend()
sns.set_style("whitegrid")
plt.tight_layout()
plt.show()
