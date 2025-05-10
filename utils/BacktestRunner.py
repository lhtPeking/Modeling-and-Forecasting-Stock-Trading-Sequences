from . import basicStrategy

class BacktestRunner:
    def __init__(self, strategy_class, df, **kwargs):
        self.strategy = strategy_class(df, **kwargs) # self.strategy是一个实例化对象

    def run(self, plot=True):
        result = self.strategy.strategy()
        metrics = self.strategy.evaluate_performance()
        print(metrics.to_string(index=False))
        if plot:
            self.strategy.plot_results()
        return result, metrics
