# 📊 Stock Sequence Modeling

A comprehensive framework for analyzing and modeling stock trading sequences using feature extraction, rule-based and deep learning strategies, clustering, and advanced evaluation methods. This project integrates both classic quantitative models and modern machine learning to build robust trading systems.

---

## 📁 Project Structure & Module Mapping

| Feature / Functionality                     | Python File / Notebook              | Description |
|---------------------------------------------|-------------------------------------|-------------|
| **Basic Financial Feature Extraction**      | `utils/basicFeature.py`             | Computes AvgReturn, Volatility, Sharpe, MDD, Skewness, Kurtosis, etc. |
| **Advanced Technical Indicators**           | `utils/advancedFeature.py`          | Computes Hurst, MACD, RSI, Bollinger Penetration, Volume-Price Correlation, etc. |
| **Feature Visualization**                   | `notebooks/Features_and_Vis.ipynb`  | Jupyter notebook for plotting basic and advanced features |
| **Rule-based Trading Strategies**           | `utils/basicStrategy.py`            | Includes moving average, MACD, RSI, Bollinger, momentum, turtle, pair trading strategies |
| **Composite Strategy (Multi-Strategy)**     | `utils/basicStrategy.py`            | Blends multiple strategies with static feature-based weight adjustments |
| **LSTM-Based Forecasting Strategy**         | `utils/timeSeriesStrategy.py`       | Implements LSTM model for time series prediction and signal generation |
| **Strategy Evaluation Utilities**           | `utils/eval.py`                     | Provides metrics for performance evaluation (returns, drawdown, Sharpe, etc.) |
| **Backtesting Runner**                      | `utils/BacktestRunner.py`           | Helper class to run strategies across multiple CSVs |
| **Clustering (KMeans, Graph, DTW)**         | `utils/cluster.py`                  | Groups stocks using static/dynamic similarity measures |
| **Strategy Simulation**                     | `notebooks/strategy.ipynb`          | Simulates and visualizes strategy performance |
| **Clustering Analysis**                     | `notebooks/clustering.ipynb`        | Visualizes KMeans and graph-based clustering results |
| **LSTM Evaluation & Metrics**               | `notebooks/evaluate.ipynb`          | Trains LSTM and analyzes trading results |
| **Misc Utilities**                          | `utils/misc.py`                     | Loads datasets, computes price panels, etc. |

---

## 🧠 Implemented Strategies

- **Dual Moving Average (DMA)**
- **MACD Crossover Strategy**
- **RSI Threshold Strategy**
- **Bollinger Band Reversion**
- **Momentum Top-N Strategy**
- **Pair Trading Strategy**
- **Turtle Trading Strategy**
- **Composite Strategy with Feature-Based Weighting**
- **LSTM-based Time Series Prediction**

---

## 🧪 Data Requirements

Place historical daily stock CSV files under the `./time-series-data/` directory.  
Each CSV should be named like `AAPL_data.csv` and include the following columns:

```csv
Date, Open, High, Low, Close, Volume
```

## Environment Setup
```
conda env create -f environment.yml
conda activate stock-modeling
```