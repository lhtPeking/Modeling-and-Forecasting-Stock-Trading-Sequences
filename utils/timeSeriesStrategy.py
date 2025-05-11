import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TradingStrategy:
    def __init__(self, stock_pool):
        """
        初始化参数：
        stock_pool: 允许交易的股票代码列表
        """
        self.stock_pool = stock_pool
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self):
        """构建简单LSTM模型"""
        class SimpleLSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                self.fc = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        self.model = SimpleLSTM(input_dim=9, hidden_dim=32, output_dim=1).to(self.device)

    def _create_features(self, df):
        """创建技术指标特征"""
        df = df.copy()
        df['ma'] = ta.trend.sma_indicator(df['Close'], window=5)
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        df['macd'] = ta.trend.macd_diff(df['Close'])
        df['volume_roc'] = ta.momentum.roc(df['Volume'])

        df = df.replace([np.inf, -np.inf], np.nan)  # ← 关键
        df = df.dropna()  # 删除无法计算的前几行
        return df


    def preprocess_data(self, raw_data):
        """
        数据预处理
        """
        X, y = [], []
        for stock in self.stock_pool:
            data = pd.DataFrame(raw_data[stock], columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Name'])
            data = self._create_features(data)
            features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'ma', 'rsi', 'macd', 'volume_roc']].fillna(0)
            labels = data['Close'].shift(-1).fillna(method='ffill')

            scaled = self.scaler.fit_transform(features)
            for i in range(len(scaled) - self.sequence_length):
                X.append(scaled[i:i+self.sequence_length])
                y.append(labels.iloc[i+self.sequence_length])

        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def _train_model(self, X_train, y_train):
        """训练模型"""
        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        self.model.train()
        for epoch in range(100):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                output = self.model(xb)
                loss = loss_fn(output, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def _predict(self, X_test):
        """预测"""
        self.model.eval()
        with torch.no_grad():
            X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            preds = self.model(X_test).cpu().numpy()
        return preds

    def generate_strategy(self, portfolio, date, real_value, next_trading_date=None):
        """
        生成每日交易策略，根据当前日期和下一个交易日的日期间隔调整策略
        :param 
        portfolio: 当前投资组合字典，包含以下字段：
        {
            'cash': 当前现金余额 float,
            'holdings': 目前持股信息 {stock: shares}, 
            'transaction_log': 历史交易记录 []
        }
        date: 需要决策的日期k, str
        real_value: 包含前面k-1天股票的真实开盘收盘价，最高价和最低价->Dict[str, List[List[Any]]]
        {
            'AAPL':[[Date,Open,High,Low,Close,Volume,Name], [Date,Open,High,Low,Close,Volume,Name], ......]
        }
        next_trading_date: 下一个交易日的日期，决定是短期还是长期策略
        :return: 交易策略列表->List[Dict[str, Dict[str, Any]]]
        [
            {'AAPL': {'action1': 'buy', 'shares1': 100, 'action2':'sell', 'shares2':50}},
            {'MSFT': {'action1': 'none', 'shares1': 0, 'action2': 'none', 'shares2': 50}}
        ]
        """

        max_shares_per_trade = 10000
        max_position_value_ratio = 0.2
        max_short_ratio = 0.5

        total_asset = portfolio['cash']
        strategy_list = []

        for stock in self.stock_pool:
            hist = real_value.get(stock, [])
            if len(hist) < self.sequence_length + 1:
                continue

            df = pd.DataFrame(hist, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Name'])
            df = self._create_features(df)
            features = df[['Open', 'High', 'Low', 'Close', 'Volume', 'ma', 'rsi', 'macd', 'volume_roc']].fillna(0)
            if len(features) < self.sequence_length:
                continue

            last_seq = features.iloc[-self.sequence_length:].values
            last_seq_scaled = self.scaler.transform(last_seq).reshape(1, self.sequence_length, -1)

            pred_price = self._predict(last_seq_scaled)[0][0]
            current_price = df['Close'].iloc[-1]

            direction = 'buy' if pred_price > current_price * 1.01 else 'sell' if pred_price < current_price * 0.99 else 'none'
            shares = min(int((total_asset * max_position_value_ratio) / current_price), max_shares_per_trade)

            if direction == 'none' or shares <= 0:
                continue

            strategy_list.append({
                stock: {
                    'action1': direction,
                    'shares1': shares,
                    'action2': 'none',
                    'shares2': 0
                }
            })

            if len(strategy_list) >= 6:
                break

        return strategy_list
