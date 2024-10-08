import yfinance as yf
import matplotlib.pyplot as plt

Z_THRESH = 3
PERIODS = [30, 60, 90]
TICKER_SYMBOL = "AMD"
START_DATE = '2024-1-1'
END_DATE = '2024-7-12'
def fetch_data(ticker_symbol, start_date, end_date):
    """Fetches historical data for a given ticker symbol."""
    ticker_data = yf.Ticker(ticker_symbol)
    return ticker_data.history(period='1d', start=start_date, end=end_date)
def calculate_z_scores(close_prices, periods):
    """Calculates Z-scores for given periods."""
    z_scores_dict = {}
    for period in periods:
        # Calculate the rolling mean for the given period
        rolling_mean = close_prices.rolling(window=period).mean()
        # Calculate the rolling standard deviation for the given period
        rolling_std = close_prices.rolling(window=period).std()
        # Compute the Z-scores for the close prices
        z_scores = (close_prices - rolling_mean) / rolling_std
        # Store the Z-scores in the dictionary with the period as the key
        z_scores_dict[period] = z_scores
    return z_scores_dict
def plot_data(close_prices, z_scores_data):
    """Plots close prices and z-scores."""
    # Create subplots for close prices and Z-scores
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(20, 8))
    # Plot the close prices on the first subplot
    ax1.plot(close_prices.index, close_prices, label='Close Prices')
    for period, z_scores in z_scores_data.items():
        # Plot the Z-scores on the second subplot for each period
        ax2.plot(z_scores.index, z_scores, label=f'Z-Scores {period} days', alpha=0.7)
        # If the period is the first in the list, plot buy/sell signals on the first subplot
        if period == PERIODS[0]:
            buy_signals = (z_scores < -Z_THRESH)
            sell_signals = (z_scores > Z_THRESH)
            ax1.plot(close_prices[buy_signals].index, close_prices[buy_signals], 'o', color='g', label='Buy Signal')
            ax1.plot(close_prices[sell_signals].index, close_prices[sell_signals], 'o', color='r', label='Sell Signal')
    # Set the y-label and legend for the close prices subplot
    ax1.set_ylabel('Close Prices')
    ax1.legend(loc="upper left")
    ax1.grid(True)
    # Draw horizontal lines indicating the Z-score thresholds on the Z-scores subplot
    ax2.axhline(-Z_THRESH, color='red', linestyle='--')
    ax2.axhline(Z_THRESH, color='red', linestyle='--')
    # Set the y-label and legend for the Z-scores subplot
    ax2.set_ylabel('Z-Scores')
    ax2.legend(loc="upper left")
    ax2.grid(True)
    # Set the main title for the entire plot
    plt.suptitle(f'{TICKER_SYMBOL} Close Prices and Z-Scores {Z_THRESH} Treshold')
    # Display the plots
    plt.show()
ticker_data = fetch_data(TICKER_SYMBOL, START_DATE, END_DATE)
z_scores_data = calculate_z_scores(ticker_data['Close'], PERIODS)
plot_data(ticker_data['Close'], z_scores_data)
plt.show()
