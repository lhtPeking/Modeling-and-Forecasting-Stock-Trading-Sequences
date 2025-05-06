import pandas as pd
import numpy as np
import os
import glob
from collections import defaultdict
from datetime import datetime, timedelta
import random
from TradingStrategy import TradingStrategy
from tqdm import tqdm
# 设置随机种子以确保可重复性
random.seed(42)

def load_stock_data(data_dir):
    """
    加载股票数据
    :param data_dir: 股票数据目录
    :return: 股票数据字典
    """
    all_data = {}
    # 获取目录中所有CSV文件
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    for file_path in csv_files:
        # 从文件名提取股票代码
        stock_code = os.path.basename(file_path).split('_')[0]
        # 读取CSV数据
        df = pd.read_csv(file_path, parse_dates=['Date'])
        # 将数据存储到字典中
        all_data[stock_code] = df
    
    return all_data

def prepare_test_data(all_data, start_date, end_date):
    """
    准备测试数据
    :param all_data: 所有股票数据的字典
    :param start_date: 开始日期
    :param end_date: 结束日期
    :return: 测试数据
    """
    test_data = {}
    
    for stock_code, df in all_data.items():
        # 筛选日期范围内的数据
        mask = (df['Date'] >= start_date) & (df['Date'] < end_date)
        filtered_data = df.loc[mask]
        
        if not filtered_data.empty:
            # 将数据转换为列表格式，匹配TradingStrategy需要的输入格式
            data_list = []
            for _, row in filtered_data.iterrows():
                data_list.append([
                    row['Date'], 
                    row['Open'], 
                    row['High'], 
                    row['Low'], 
                    row['Close'], 
                    row['Volume'], 
                    stock_code
                ])
            test_data[stock_code] = data_list
    
    return test_data

def generate_trading_dates(start_date, num_days, min_gap=1, max_gap=100):
    """
    随机生成交易日序列
    :param start_date: 起始日期
    :param num_days: 交易日数量
    :param min_gap: 最小间隔天数
    :param max_gap: 最大间隔天数
    :return: 交易日列表
    """
    trading_dates = []
    current_date = start_date
    
    for _ in range(num_days):
        trading_dates.append(current_date)
        # 随机生成下一个交易日的间隔天数
        gap = random.randint(min_gap, max_gap)
        current_date = current_date + timedelta(days=gap)
    
    return trading_dates

def evaluate_strategy(strategy_class, all_stock_data, initial_cash=1000000, num_trading_days=30):
    """
    评测函数实现
    :param strategy_class: 学生实现的策略类
    :param all_stock_data: 所有股票的历史数据
    :param initial_cash: 初始资金（默认100万）
    :param num_trading_days: 评估的交易日数量
    """
    # 初始化交易记录
    portfolio = {
        'cash': initial_cash,
        'holdings': defaultdict(int),  # {stock: shares}，正值表示多头持仓，负值表示空头持仓
        'transaction_log': []
    }

    # 风险控制参数
    max_shares_per_trade = 10000  # 单次交易最大股数
    max_position_value_ratio = 0.2  # 单个股票最大持仓价值占总资产的比例
    max_short_ratio = 0.5  # 最大卖空比例

    # 生成随机交易日序列
    start_date = datetime(2011, 1, 1)  # 根据数据集的实际时间范围调整
    trading_dates = generate_trading_dates(start_date, num_trading_days)
    # 逐日模拟交易
    for i in tqdm(range(num_trading_days-1)):  # 最后一天不需要生成策略
        current_date = trading_dates[i]
        next_trading_date = trading_dates[i+1]
        
        # 准备当前日期的历史数据
        test_data = prepare_test_data(all_stock_data, start_date, current_date)
        
        # 计算当前总资产价值
        total_portfolio_value = portfolio['cash']
        for stock, shares in portfolio['holdings'].items():
            if stock in test_data and test_data[stock]:
                stock_price = test_data[stock][-1][4]  # 收盘价
                if shares > 0:  # 多头持仓
                    total_portfolio_value += stock_price * shares
                elif shares < 0:  # 空头持仓
                    total_portfolio_value -= stock_price * (-shares)

        # 如果start_date之前用户持有股票，则系统基于start_date前一天的收盘价进行平仓
        if current_date > start_date and i > 0:
            # 平仓操作，基于current_date前一个交易日的收盘价，
            for stock, shares in list(portfolio['holdings'].items()):
                if shares == 0:
                    continue
                
                if stock in test_data and test_data[stock]:
                    closing_price = test_data[stock][-2][4]  # 收盘价
                    if shares > 0:  # 多头平仓
                        portfolio['cash'] += closing_price * shares
                        portfolio['holdings'][stock] = 0
                    elif shares < 0:  # 空头平仓
                        portfolio['cash'] -= closing_price * (-shares)
                        portfolio['holdings'][stock] = 0

                    portfolio['holdings'] = {}

        # 生成当日策略
        strategy_list = strategy_class.generate_strategy(
            portfolio=portfolio,
            date=current_date,
            real_value=test_data,
            next_trading_date=next_trading_date
        )
        
        # 将策略列表转换为字典，用于记录日志
        strategy_dict = {}
        for item in strategy_list:
            for stock, order in item.items():
                strategy_dict[stock] = order
        
        # 执行交易校验（关键边界条件处理）
        valid_orders = []
        
        # 按照策略列表的顺序处理交易（保持优先级）
        for strategy_item in strategy_list:
            # 每个策略项是一个字典，只包含一个股票代码和其交易信息
            for stock, order in strategy_item.items():
                # 边界检查1：股票池校验
                if stock not in strategy_class.stock_pool:
                    continue
                
                # 获取当前股票价格数据
                if stock not in test_data or not test_data[stock]:
                    continue
                
                stock_data = test_data[stock][-1]  # 取最近一天的数据
                current_open = stock_data[1]  # 开盘价
                current_close = stock_data[4]  # 收盘价
                
                if stock not in portfolio['holdings']:
                            portfolio['holdings'][stock] = 0

                # 处理action1（第一个操作）
                action1 = order.get('action1', 'none')
                shares1 = min(order.get('shares1', 0), max_shares_per_trade)  # 限制单次交易股数
                
                if action1 != 'none' and shares1 > 0:
                    if action1 == 'buy':
                        # 边界检查：资金充足性
                        cost = current_open * shares1
                        if cost > portfolio['cash']:
                            # 调整可购买的股数
                            shares1 = int(portfolio['cash'] / current_open)
                        
                        if shares1 <= 0:
                            continue
                            
                        cost = current_open * shares1
                        
                        # 检查持仓限制
                        new_position_value = (portfolio['holdings'][stock] + shares1) * current_open
                        if new_position_value > total_portfolio_value * max_position_value_ratio:
                            # 调整购买股数，使其符合持仓限制
                            max_allowed_shares = int((total_portfolio_value * max_position_value_ratio - 
                                                    portfolio['holdings'][stock] * current_open) / current_open)
                            shares1 = max(0, max_allowed_shares)
                            if shares1 <= 0:
                                continue
                            cost = current_open * shares1
                        
                        # 更新持仓和资金
                        portfolio['holdings'][stock] += shares1
                        portfolio['cash'] -= cost
                        valid_orders.append((stock, 'buy', shares1))
                        
                    elif action1 == 'sell':
                        # 检查是否为平仓操作（之前有多头持仓）
                        if stock in portfolio['holdings'] and portfolio['holdings'][stock] > 0:
                            # 平仓操作
                            available_shares = portfolio['holdings'][stock]
                            sell_shares = min(shares1, available_shares)
                            
                            # 更新资金和持仓
                            portfolio['cash'] += current_close * sell_shares
                            portfolio['holdings'][stock] -= sell_shares
                            valid_orders.append((stock, 'sell', sell_shares))
                        else:
                            # 卖空操作 - 限制卖空规模
                            short_value = shares1 * current_open
                            max_short_value = total_portfolio_value * max_short_ratio
                            
                            if short_value > max_short_value:
                                shares1 = int(max_short_value / current_open)
                                if shares1 <= 0:
                                    continue
                            
                            # 直接更新持仓（负值表示空头）

                            portfolio['holdings'][stock] -= shares1
                            # 卖空时增加资金
                            portfolio['cash'] += current_close * shares1
                            valid_orders.append((stock, 'short', shares1))
                
                # 处理action2（第二个操作）
                action2 = order.get('action2', 'none')
                shares2 = min(order.get('shares2', 0), max_shares_per_trade)  # 限制单次交易股数
                
                if action2 != 'none' and shares2 > 0:
                    if action2 == 'buy':
                        # 边界检查：资金充足性
                        cost = current_open * shares2
                        if cost > portfolio['cash']:
                            # 调整可购买的股数
                            shares2 = int(portfolio['cash'] / current_open)
                        
                        if shares2 <= 0:
                            continue
                        
                        cost = current_open * shares2
                        
                        # 检查持仓限制
                        new_position_value = (portfolio['holdings'][stock] + shares2) * current_open
                        if new_position_value > total_portfolio_value * max_position_value_ratio:
                            # 调整购买股数，使其符合持仓限制
                            max_allowed_shares = int((total_portfolio_value * max_position_value_ratio - 
                                                    portfolio['holdings'][stock] * current_open) / current_open)
                            shares2 = max(0, max_allowed_shares)
                            if shares2 <= 0:
                                continue
                            cost = current_open * shares2
                        
                        # 更新持仓和资金
                        portfolio['holdings'][stock] += shares2
                        portfolio['cash'] -= cost
                        valid_orders.append((stock, 'buy', shares2))
                        
                    elif action2 == 'sell':
                        # 检查是否为平仓操作（之前有多头持仓）
                        if stock in portfolio['holdings'] and portfolio['holdings'][stock] > 0:
                            # 平仓操作
                            available_shares = portfolio['holdings'][stock]
                            sell_shares = min(shares2, available_shares)
                            
                            # 更新资金和持仓
                            portfolio['cash'] += current_close * sell_shares
                            portfolio['holdings'][stock] -= sell_shares
                            valid_orders.append((stock, 'sell', sell_shares))
                        else:
                            # 卖空操作 - 限制卖空规模
                            short_value = shares2 * current_open
                            max_short_value = total_portfolio_value * max_short_ratio
                            
                            if short_value > max_short_value:
                                shares2 = int(max_short_value / current_open)
                                if shares2 <= 0:
                                    continue
                            
                            # 直接更新持仓（负值表示空头）
                            portfolio['holdings'][stock] -= shares2
                            # 卖空时增加资金
                            portfolio['cash'] += current_close * shares2
                            valid_orders.append((stock, 'short', shares2))
        
        # 记录当日交易
        portfolio['transaction_log'].append({
            'date': current_date,
            'valid_orders': valid_orders,
            'remaining_cash': portfolio['cash'],
            'holdings': dict(portfolio['holdings']),
            'strategy': strategy_dict  # 使用转换后的字典格式记录策略
        })
    
    # 在最后一个交易日平仓所有持仓
    final_date = trading_dates[-1]
    final_data = prepare_test_data(all_stock_data, start_date, final_date)
    
    # 计算最终资产
    final_value = portfolio['cash']
    valid_orders = []
    
    for stock, shares in list(portfolio['holdings'].items()):
        if shares == 0:
            continue
            
        if stock in final_data and final_data[stock]:
            final_price = final_data[stock][-1][4]  # 收盘价
            
            if shares > 0:  # 多头持仓
                # 平多仓
                final_value += final_price * shares
                valid_orders.append((stock, 'sell', shares))
                
            elif shares < 0:  # 空头持仓
                # 平空仓，计算卖空收益或损失
                short_value = -shares * final_price
                final_value -= short_value
                valid_orders.append((stock, 'cover', -shares))
    
    # 记录最后一天的平仓交易
    portfolio['transaction_log'].append({
        'date': final_date,
        'valid_orders': valid_orders,
        'remaining_cash': final_value,
        'holdings': {},  # 所有持仓已平仓
        'strategy': {}
    })
    
    return final_value, portfolio['transaction_log']

def main():
    # 加载股票数据
    data_dir = "./time-series-data"
    all_stock_data = load_stock_data(data_dir)
    
    # 获取股票池
    stock_pool = list(all_stock_data.keys())
    
    # 创建交易策略实例
    strategy = TradingStrategy(stock_pool)
    
    # 评估策略
    final_value, transaction_log = evaluate_strategy(strategy, all_stock_data)
    
    # 输出评估结果
    print(f"初始资金: 1,000,000")
    print(f"最终资产: {final_value:.2f}")
    print(f"收益率: {(final_value-1000000)/1000000*100:.2f}%")
    
    # 输出交易记录摘要
    print("\n交易记录摘要:")
    for i, log in enumerate(transaction_log):
        print(f"日期: {log['date'].strftime('%Y-%m-%d')}")
        print(f"  交易: {len(log['valid_orders'])} 笔")
        print(f"  现金余额: {log['remaining_cash']:.2f}")
        print(f"  持仓股票数: {len(log['holdings'])}")
        print(f"  策略: {log['strategy']}")
        print("-" * 40)
        # if i < 5 or i >= len(transaction_log) - 5:  # 只显示前5天和后5天

        # elif i == 5:
        #     print("... ...")

if __name__ == "__main__":
    main()