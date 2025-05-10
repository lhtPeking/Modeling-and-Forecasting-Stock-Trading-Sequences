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
    
    def _build_model(self):
        
    
    def _create_features(self):
        """创建技术指标特征"""

    

    def _train_model(self):
        
    
    def preprocess_data(self):
        """
        数据预处理
        """

    
    def _predict(self):
        """预测"""
    
    
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

        该函数的输入请不要更改，返回值要符合上面的格式，以便于后续的回测和评估
        """

        
        return strategy_list