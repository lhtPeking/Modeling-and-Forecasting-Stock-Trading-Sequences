import pandas as pd
import numpy as np
import os

class basicFeature:
    @staticmethod
    def load_data(path):
        df = pd.read_csv(path)
        exclude_cols = ['Date', 'Name']
        target_cols = [col for col in df.columns if col not in exclude_cols]
        for col in target_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df[target_cols] = df[target_cols].fillna(df[target_cols].rolling(window=5, min_periods=1).mean())
        df.dropna(subset=['Close'], inplace=True)
        return df

    @staticmethod
    def compute_AvgReturn(path):
        df = basicFeature.load_data(path)
        returns = df['Close'].pct_change().dropna()
        return returns.mean()

    @staticmethod
    def compute_Volatility(path):
        df = basicFeature.load_data(path)
        returns = df['Close'].pct_change().dropna()
        return returns.std()

    @staticmethod
    def compute_SharpeRatio(path, risk_free_rate=0.0):
        df = basicFeature.load_data(path)
        returns = df['Close'].pct_change().dropna()
        excess_return = returns - risk_free_rate
        if returns.std() == 0:
            return np.nan
        return excess_return.mean() / returns.std()

    @staticmethod
    def compute_MDD(path):
        df = basicFeature.load_data(path)
        cum_returns = (1 + df['Close'].pct_change().fillna(0)).cumprod()
        rolling_max = cum_returns.cummax()
        drawdown = cum_returns / rolling_max - 1
        return drawdown.min()

    @staticmethod
    def compute_Skewness(path):
        df = basicFeature.load_data(path)
        returns = df['Close'].pct_change().dropna()
        return returns.skew()

    @staticmethod
    def compute_Kurtosis(path):
        df = basicFeature.load_data(path)
        returns = df['Close'].pct_change().dropna()
        return returns.kurt()
    
    @staticmethod
    def integrate_all_features(path):
        df = basicFeature.load_data(path)
        features = {
            'AvgReturn': basicFeature.compute_AvgReturn(path),
            'Volatility': basicFeature.compute_Volatility(path),
            'SharpeRatio': basicFeature.compute_SharpeRatio(path),
            'MDD': basicFeature.compute_MDD(path),
            'Skewness': basicFeature.compute_Skewness(path),
            'Kurtosis': basicFeature.compute_Kurtosis(path)
        }
        stock_name = os.path.basename(path).split('_')[0]
        return pd.DataFrame(features, index=[stock_name])
