import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        
class macd_strategy(BaseStrategy):
    def __init__(self, df):
        super().__init__(df)
    
    def strategy(self):
        df = self.df
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        # MACD核心计算
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['DIF'] = df['EMA_12'] - df['EMA_26']
        df['DEA'] = df['DIF'].ewm(span=9).mean()
        df['Signal'] = np.where(df['DIF'] > df['DEA'], 1, -1)

        # 收益计算
        df['MarketReturn'] = df['Close'].pct_change()
        df['StrategyReturn'] = df['MarketReturn'] * df['Signal'].shift(1)
        df['CumulativeStrategy'] = (1 + df['StrategyReturn'].fillna(0)).cumprod()
        return df

class bollinger_strategy(BaseStrategy):
    def __init__(self, df):
        super().__init__(df)
    
    def strategy(self):
        df = self.df
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        df['MA_20'] = df['Close'].rolling(20).mean()
        df['STD_20'] = df['Close'].rolling(20).std()
        df['Upper'] = df['MA_20'] + 2 * df['STD_20']
        df['Lower'] = df['MA_20'] - 2 * df['STD_20']

        df['Signal'] = np.where(df['Close'] < df['Lower'], 1,
                          np.where(df['Close'] > df['Upper'], -1, 0))

        df['MarketReturn'] = df['Close'].pct_change()
        df['StrategyReturn'] = df['MarketReturn'] * df['Signal'].shift(1)
        df['CumulativeStrategy'] = (1 + df['StrategyReturn'].fillna(0)).cumprod()
        return df
    
    def plot_results(self):
        """
        Plot price with Bollinger Bands and strategy equity curve
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Price and Bollinger Bands
        axes[0].plot(self.df.index, self.df['Close'], label='Close Price', color='black')
        axes[0].plot(self.df.index, self.df['MA_20'], label='MA 20', color='blue')
        axes[0].plot(self.df.index, self.df['Upper'], label='Upper Band', linestyle='--', color='red')
        axes[0].plot(self.df.index, self.df['Lower'], label='Lower Band', linestyle='--', color='green')
        axes[0].fill_between(self.df.index, self.df['Lower'], self.df['Upper'], color='gray', alpha=0.1)

        buy_signals = self.df[self.df['Signal'].diff() == 2].index
        sell_signals = self.df[self.df['Signal'].diff() == -2].index
        axes[0].scatter(buy_signals, self.df.loc[buy_signals, 'Close'], marker='^', label='Buy Signal')
        axes[0].scatter(sell_signals, self.df.loc[sell_signals, 'Close'], marker='v', label='Sell Signal')

        axes[0].set_title('Bollinger Bands & Trading Signals')
        axes[0].set_ylabel('Price')
        axes[0].legend()

        # Equity Curve
        axes[1].plot(self.df.index, self.df['CumulativeStrategy'], label='Bollinger Strategy Return')
        axes[1].set_title('Strategy Equity Curve')
        axes[1].set_ylabel('Cumulative Return')
        axes[1].legend()

        plt.xlabel('Date')
        plt.tight_layout()
        plt.show()

class rsi_strategy(BaseStrategy):
    def __init__(self, df):
        super().__init__(df)
    
    def strategy(self):
        df = self.df
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        df['Signal'] = np.where(df['RSI'] < 30, 1,
                          np.where(df['RSI'] > 70, -1, 0))

        df['MarketReturn'] = df['Close'].pct_change()
        df['StrategyReturn'] = df['MarketReturn'] * df['Signal'].shift(1)
        df['CumulativeStrategy'] = (1 + df['StrategyReturn'].fillna(0)).cumprod()
        return df

class pair_trading_strategy(BaseStrategy):
    def __init__(self, df):
        super().__init__(df)
    
    def strategy(self):
        df = self.df
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        # 计算价差与z-score
        df['Spread'] = df['StockA'] - df['StockB']
        df['Zscore'] = (df['Spread'] - df['Spread'].rolling(30).mean()) / df['Spread'].rolling(30).std()

        # 策略信号
        df['Signal'] = np.where(df['Zscore'] < -1.5, 1, np.where(df['Zscore'] > 1.5, -1, 0))

        # 简化收益计算（实际应分别计算StockA和StockB的对冲收益）
        df['MarketReturn'] = df['StockA'].pct_change()
        df['StrategyReturn'] = df['MarketReturn'] * df['Signal'].shift(1)
        df['CumulativeStrategy'] = (1 + df['StrategyReturn'].fillna(0)).cumprod()
        return df

class momentum_topN_strategy(BaseStrategy):
    def __init__(self, df, lookback=20, top_n=3):
        super().__init__(df)
        self.lookback = lookback
        self.top_n = top_n

    def strategy(self):
        df = self.df.copy()

        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("Expected MultiIndex with levels ['Date', 'Ticker'].")

        df = df.sort_index()

        # 动量与信号
        df['Momentum'] = df.groupby(level=1)['Close'].pct_change(self.lookback)
        df['Rank'] = df.groupby(level=0)['Momentum'].rank(method='first', ascending=False)
        df['Signal'] = (df['Rank'] <= self.top_n).astype(int)

        # 收益
        df['MarketReturn'] = df.groupby(level=1)['Close'].pct_change()
        df['StrategyReturn'] = df['MarketReturn'] * df['Signal'].shift(1)

        # 组合级别等权合成
        combo_df = df.groupby(level=0).agg({
            'Close': 'mean',
            'Signal': 'mean',
            'StrategyReturn': 'mean'
        })

        combo_df['CumulativeStrategy'] = (1 + combo_df['StrategyReturn'].fillna(0)).cumprod()

        # 保存结果
        self.df = combo_df                   # 给 BaseStrategy.evaluate_performance 用
        self.df_full = df                   # 给 plot_results 用
        return df

    def plot_results(self, show_ticker_list=None):
        """
        只可视化多支股票的持仓热力图（Top-N Signal），横坐标仅显示日期
        """
        import matplotlib.dates as mdates

        fig, ax = plt.subplots(1, 1, figsize=(14, 8), sharex=True)

        # 解构 signal 矩阵
        signal_df = self.df_full['Signal'].unstack(level=1).fillna(0)

        # 可选筛选股票
        if show_ticker_list:
            missing = set(show_ticker_list) - set(signal_df.columns)
            if missing:
                print(f"Warning: Some tickers not in data: {missing}")
            signal_df = signal_df[signal_df.columns.intersection(show_ticker_list)]

        # 绘图
        sns.heatmap(signal_df.T, cmap="YlGnBu", cbar=True, ax=ax)

        # 优化横坐标日期显示
        ax.set_xticks(np.linspace(0, len(signal_df.index)-1, 10, dtype=int))  # 每隔一定步长打标签
        ax.set_xticklabels(
            [signal_df.index[i].strftime('%Y-%m-%d') for i in ax.get_xticks()],
            rotation=45,
            ha='right'
        )

        ax.set_title('Top-N Stock Selection Heatmap')
        ax.set_ylabel('Stock')
        ax.set_xlabel('Date')

        plt.tight_layout()
        plt.show()





class turtle_strategy(BaseStrategy):
    def __init__(self, df):
        super().__init__(df)
    
    def strategy(self):
        df = self.df
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        df['TR'] = np.maximum(df['High'] - df['Low'],
                      np.maximum(abs(df['High'] - df['Close'].shift()),
                                 abs(df['Low'] - df['Close'].shift())))
        df['ATR'] = df['TR'].rolling(20).mean()

        df['High20'] = df['High'].rolling(20).max()
        df['Low20'] = df['Low'].rolling(20).min()

        df['Signal'] = 0
        df.loc[df['Close'] > df['High20'].shift(1), 'Signal'] = 1
        df.loc[df['Close'] < df['Low20'].shift(1), 'Signal'] = -1

        df['MarketReturn'] = df['Close'].pct_change()
        df['StrategyReturn'] = df['MarketReturn'] * df['Signal'].shift(1)
        df['CumulativeStrategy'] = (1 + df['StrategyReturn'].fillna(0)).cumprod()
        return df

class composite_strategy(BaseStrategy):
    def __init__(self, df, static_features=None):
        super().__init__(df)
        self.sub_strategies = []
        self.feature_weights = {}
        self.static_features = static_features or {}

    def add_strategy(self, strategy_cls, weight=1.0, **kwargs):
        strat = strategy_cls(self.df.copy(), **kwargs)
        self.sub_strategies.append((strat, weight))

    def compute_static_weights(self):
        """
        根据静态特征调整策略整体权重
        """
        avg_return = self.static_features.get("AvgReturn", 0.0005)
        volatility = self.static_features.get("Volatility", 0.02)
        sharpe = self.static_features.get("SharpeRatio", 1.0)

        # 构造静态权重因子（加权调节）
        w_ret = 1 + 5 * avg_return
        w_vol = 1 - 3 * volatility
        w_sharpe = 1 + 0.5 * sharpe

        combined_weight = w_ret * w_vol * w_sharpe
        self.feature_weights = {"static_weight": max(0.1, min(combined_weight, 2.0))}
        return self.feature_weights

    def strategy(self):
        df = self.df
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        self.compute_static_weights()

        signal_sum = 0
        total_weight = 0

        for strat, base_weight in self.sub_strategies:
            strat_df = strat.strategy()
            weight = base_weight * self.feature_weights.get("static_weight", 1.0)
            signal_sum += weight * strat_df['Signal']
            total_weight += weight

        df['Signal'] = np.sign(signal_sum / total_weight)
        df['MarketReturn'] = df['Close'].pct_change()
        df['StrategyReturn'] = df['MarketReturn'] * df['Signal'].shift(1)
        df['CumulativeStrategy'] = (1 + df['StrategyReturn'].fillna(0)).cumprod()
        self.df = df
        return df
