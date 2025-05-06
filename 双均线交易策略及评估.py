import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(filepath):
    """
    Load CSV file, parse dates, and set index
    """
    df = pd.read_csv(filepath, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df


def dual_ma_strategy(df, short_window=10, long_window=30):
    """
    Generate dual moving average signals and strategy returns
    """
    df = df.copy()
    df[f'SMA_{short_window}'] = df['Close'].rolling(window=short_window).mean()
    df[f'SMA_{long_window}']  = df['Close'].rolling(window=long_window).mean()
    df['Signal'] = 0
    df.loc[df[f'SMA_{short_window}'] > df[f'SMA_{long_window}'], 'Signal'] = 1
    df.loc[df[f'SMA_{short_window}'] < df[f'SMA_{long_window}'], 'Signal'] = -1
    # Strategy returns
    df['MarketReturn'] = df['Close'].pct_change()
    df['StrategyReturn'] = df['MarketReturn'] * df['Signal'].shift(1)
    df['CumulativeStrategy'] = (1 + df['StrategyReturn'].fillna(0)).cumprod()
    return df


def evaluate_performance(df):
    """
    Calculate performance metrics for the dual MA strategy only
    """
    total_ret = df['CumulativeStrategy'].iloc[-1] - 1
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25
    cagr = df['CumulativeStrategy'].iloc[-1]**(1/years) - 1
    ann_vol = df['StrategyReturn'].std() * np.sqrt(252)
    sharpe = (df['StrategyReturn'].mean()*252) / ann_vol
    rolling_max = df['CumulativeStrategy'].cummax()
    max_dd = (df['CumulativeStrategy']/rolling_max - 1).min()

    metrics = pd.DataFrame({
        'Metric': ['Total Return', 'CAGR', 'Annual Volatility', 'Sharpe Ratio', 'Max Drawdown'],
        'Value': [total_ret, cagr, ann_vol, sharpe, max_dd]
    })
    return metrics


def plot_results(df, short_window=10, long_window=30):
    """
    Plot price with MA signals and strategy equity curve
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Price and MA signals
    axes[0].plot(df.index, df['Close'], label='Close Price')
    axes[0].plot(df.index, df[f'SMA_{short_window}'], label=f'{short_window}-day SMA')
    axes[0].plot(df.index, df[f'SMA_{long_window}'], label=f'{long_window}-day SMA')
    buy_signals = df[df['Signal'].diff() == 2].index
    sell_signals = df[df['Signal'].diff() == -2].index
    axes[0].scatter(buy_signals, df.loc[buy_signals, 'Close'], marker='^', label='Buy')
    axes[0].scatter(sell_signals, df.loc[sell_signals, 'Close'], marker='v', label='Sell')
    axes[0].set_title('Price and MA Signals')
    axes[0].set_ylabel('Price')
    axes[0].legend()

    # Equity curve of strategy
    axes[1].plot(df.index, df['CumulativeStrategy'], label='Dual MA Strategy')
    axes[1].set_title('Strategy Equity Curve')
    axes[1].set_ylabel('Cumulative Return')
    axes[1].legend()
    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    filepath = './time-series-data/BA_2006-01-01_to_2018-01-01.csv'
    df = load_data(filepath)
    df_signal = dual_ma_strategy(df, short_window=10, long_window=30)
    metrics = evaluate_performance(df_signal)
    print("Strategy Performance Metrics:")
    print(metrics.to_string(index=False, float_format="{:.4f}".format))
    plot_results(df_signal)
