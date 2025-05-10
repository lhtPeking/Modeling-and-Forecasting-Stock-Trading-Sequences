import os
import pandas as pd
from utils.basicFeature import basicFeature  # 确保你路径正确

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
