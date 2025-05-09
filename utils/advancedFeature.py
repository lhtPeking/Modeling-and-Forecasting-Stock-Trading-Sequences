import pandas as pd
import numpy as np
import os

class advancedFeature:
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
    def compute_Hurst(path):
        df = advancedFeature.load_data(path)
        ts = df['Close'].dropna()
        lags = range(2, 100)
        tau = [np.std(ts.diff(lag).dropna()) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]

    @staticmethod
    def compute_MACD_mean(path):
        df = advancedFeature.load_data(path)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        return macd.mean()

    @staticmethod
    def compute_RSI14_mean(path):
        df = advancedFeature.load_data(path)
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.mean()

    @staticmethod
    def compute_Bollinger_penetration_rate(path):
        df = advancedFeature.load_data(path)
        ma20 = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        penetration = ((df['Close'] > upper) | (df['Close'] < lower)).sum()
        total = len(df)
        return penetration / total
    
    @staticmethod
    def compute_volume_price_corr(path):
        df = advancedFeature.load_data(path)
        return df[['Close', 'Volume']].corr().iloc[0, 1]

    @staticmethod
    def compute_overnight_jump_rate(path, threshold=0.01):
        df = advancedFeature.load_data(path)
        df['PrevClose'] = df['Close'].shift(1)
        df['OvernightReturn'] = (df['Open'] - df['PrevClose']) / df['PrevClose']
        jump_rate = (df['OvernightReturn'].abs() > threshold).mean()
        return jump_rate

    @staticmethod
    def compute_weekday_return_std(path):
        df = advancedFeature.load_data(path)
        df['Return'] = df['Close'].pct_change()
        df['Weekday'] = pd.to_datetime(df['Date']).dt.weekday
        return df.groupby('Weekday')['Return'].std().mean()

    @staticmethod
    def compute_monthly_volatility_autocorr(path):
        df = advancedFeature.load_data(path)
        df['Return'] = df['Close'].pct_change()
        df['Month'] = pd.to_datetime(df['Date']).dt.to_period('M')
        monthly_vol = df.groupby('Month')['Return'].std()
        return monthly_vol.autocorr()

    @staticmethod
    def integrate_advanced_features(path):
        df = advancedFeature.load_data(path)
        features = {
            'Hurst': advancedFeature.compute_Hurst(path),
            'MACD_mean': advancedFeature.compute_MACD_mean(path),
            'RSI14_mean': advancedFeature.compute_RSI14_mean(path),
            'BollingerPenRate': advancedFeature.compute_Bollinger_penetration_rate(path),
            'VolumePriceCorr': advancedFeature.compute_volume_price_corr(path),
            'OvernightJumpRate': advancedFeature.compute_overnight_jump_rate(path),
            'WeekdayReturnStd': advancedFeature.compute_weekday_return_std(path),
            'MonthlyVolAutocorr': advancedFeature.compute_monthly_volatility_autocorr(path)
        }
        stock_name = os.path.basename(path).split('_')[0]
        return pd.DataFrame(features, index=[stock_name])