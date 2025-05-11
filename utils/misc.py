import os
import pandas as pd
import glob
from utils.basicFeature import basicFeature

def load_multi_stock_data(stock_list, folder_path="./time-series-data"):
    """
    加载多个股票数据并拼接为MultiIndex DataFrame（适用于多股票轮动策略）
    
    参数：
        stock_list: ['AAPL', 'MSFT', 'GOOG']
        folder_path: 存放CSV文件的文件夹路径

    返回：
        MultiIndex DataFrame（index为[Date, Ticker]）
    """
    df_dict = {}

    for stock in stock_list:
        filename = f"{stock}_2006-01-01_to_2018-01-01.csv"
        path = os.path.join(folder_path, filename)
        try:
            df = basicFeature.load_data(path)
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
            df_dict[stock] = df
        except Exception as e:
            print(f"Error loading {stock}: {e}")

    # 拼接所有股票数据，形成 MultiIndex: (Ticker, Date)
    combined_df = pd.concat(df_dict, names=["Ticker"])
    combined_df = combined_df.reset_index().set_index(["Date", "Ticker"]).sort_index()

    return combined_df

def load_static_features(stock_list, folder="./time-series-data"):
    features = []
    for stock in stock_list:
        path = os.path.join(folder, f"{stock}_2006-01-01_to_2018-01-01.csv")
        try:
            f = basicFeature.integrate_all_features(path)
            features.append(f)
        except Exception as e:
            print(f"{stock} failed: {e}")
    return pd.concat(features)

def load_price_panel(stock_list=None, folder="./time-series-data/", price_col="Close"):
    """
    加载多个股票的收盘价数据，返回以日期为索引、股票为列的 DataFrame。
    
    参数：
        stock_list: 股票代码列表（可选）。若为 None，将从目录自动提取。
        folder: CSV 文件所在目录。
        price_col: 价格字段名，通常为 "Close"、"Adj Close" 等。
    
    返回：
        price_df: DataFrame，行是日期，列是股票名，值是指定价格字段。
    """
    all_prices = {}

    # 自动抓取目录下所有股票名
    if stock_list is None:
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        stock_list = [os.path.basename(f).split("_")[0] for f in csv_files]

    for stock in stock_list:
        path = os.path.join(folder, f"{stock}_2006-01-01_to_2018-01-01.csv")
        try:
            df = pd.read_csv(path, parse_dates=['Date'])
            df = df[['Date', price_col]].dropna()
            df.set_index('Date', inplace=True)
            all_prices[stock] = df[price_col]
        except Exception as e:
            print(f"[Warning] {stock} skipped: {e}")

    price_df = pd.DataFrame(all_prices)
    price_df = price_df.dropna(how='all')  # 丢弃全为NaN的日期
    return price_df
