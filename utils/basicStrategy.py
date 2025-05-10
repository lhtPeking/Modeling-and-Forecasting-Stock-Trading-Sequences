import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BaseStrategy:
    def __init__(self, df):
        self.df = df.copy() # 注意传参之前需要传load好的df而不是path
        
    def evaluate_performance(self):
        """
        Calculate performance metrics for the dual MA strategy only
        """
        total_ret = self.df['CumulativeStrategy'].iloc[-1] - 1
        days = (self.df.index[-1] - self.df.index[0]).days
        years = days / 365.25
        cagr = self.df['CumulativeStrategy'].iloc[-1]**(1/years) - 1
        ann_vol = self.df['StrategyReturn'].std() * np.sqrt(252)
        sharpe = (self.df['StrategyReturn'].mean()*252) / ann_vol
        rolling_max = self.df['CumulativeStrategy'].cummax()
        max_dd = (self.df['CumulativeStrategy']/rolling_max - 1).min()

        metrics = pd.DataFrame({
            'Metric': ['Total Return', 'CAGR', 'Annual Volatility', 'Sharpe Ratio', 'Max Drawdown'],
            'Value': [total_ret, cagr, ann_vol, sharpe, max_dd]
        })
        return metrics

    def strategy(self):
        raise NotImplementedError("You must implement gstrategy() in subclass.")

class dual_ma_strategy(BaseStrategy):
    def __init__(self, df, short_window=10, long_window=30):
        super().__init__(df)
        
        self.short_window = short_window
        self.long_window = long_window
    
    def strategy(self):
        """
        Generate dual moving average signals and strategy returns
        """
        df = self.df
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        df[f'SMA_{self.short_window}'] = df['Close'].rolling(window=self.short_window).mean()
        df[f'SMA_{self.long_window}']  = df['Close'].rolling(window=self.long_window).mean()
        df['Signal'] = 0
        df.loc[df[f'SMA_{self.short_window}'] > df[f'SMA_{self.long_window}'], 'Signal'] = 1
        df.loc[df[f'SMA_{self.short_window}'] < df[f'SMA_{self.long_window}'], 'Signal'] = -1
        # Strategy returns
        df['MarketReturn'] = df['Close'].pct_change()
        df['StrategyReturn'] = df['MarketReturn'] * df['Signal'].shift(1)
        df['CumulativeStrategy'] = (1 + df['StrategyReturn'].fillna(0)).cumprod()
        return df
    
    def plot_results(self):
        """
        Plot price with MA signals and strategy equity curve
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Price and MA signals
        axes[0].plot(self.df.index, self.df['Close'], label='Close Price')
        axes[0].plot(self.df.index, self.df[f'SMA_{self.short_window}'], label=f'{self.short_window}-day SMA')
        axes[0].plot(self.df.index, self.df[f'SMA_{self.long_window}'], label=f'{self.long_window}-day SMA')
        buy_signals = self.df[self.df['Signal'].diff() == 2].index
        sell_signals = self.df[self.df['Signal'].diff() == -2].index
        axes[0].scatter(buy_signals, self.df.loc[buy_signals, 'Close'], marker='^', label='Buy')
        axes[0].scatter(sell_signals, self.df.loc[sell_signals, 'Close'], marker='v', label='Sell')
        axes[0].set_title('Price and MA Signals')
        axes[0].set_ylabel('Price')
        axes[0].legend()

        # Equity curve of strategy
        axes[1].plot(self.df.index, self.df['CumulativeStrategy'], label='Dual MA Strategy')
        axes[1].set_title('Strategy Equity Curve')
        axes[1].set_ylabel('Cumulative Return')
        axes[1].legend()
        plt.xlabel('Date')
        plt.tight_layout()
        plt.show()
        
    