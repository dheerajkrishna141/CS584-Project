# Stock Market Data Analysis and Prediction: Using Technical Strategies

This project is part of a CS course on the **Theory and Applications of Data Mining**, focusing on implementing and optimizing stock trading strategies using historical stock data and prediction based on modeling techniques learned from the class.

The core functionality involves backtesting and optimizing **Simple Moving Average (SMA)** strategies, **rolling regression**, and evaluating performance using metrics such as **F1 Score**, **Accuracy**, **Precision** and **Recall**

---

### Project Structure

| File                    | Purpose                                                                 |
| ----------------------- | ----------------------------------------------------------------------- |
| `smaDDBacktester.py`    | Main script for strategy optimization and backtesting                   |
| `main.py`               | Real-time trading execution script using Alpaca API                     |
| `plotting.py`           | Visualizes SMA vs slope-based crossover signals for each stock          |
| `confusion_plots.py`    | Generates annotated confusion matrices and evaluation metrics per stock |
| `config.ini`            | API credentials for Alpaca (stored under `[alpaca]`)                    |
| `optimized_results.csv` | Output file containing optimized strategy results                       |
| `requirements.txt`      | Python libraries used in this project                                   |

---

### Project Structure for Pairs Trading

| File / Folder          | Purpose                                                     |
| ---------------------- | ----------------------------------------------------------- |
| `pairs_trader.ipynb`   | Main notebook: finds pairs, runs backtests, executes trades |
| `config.ini`           | Stores strategy settings and parameters                     |
| `requirements.txt`     | Lists required Python libraries                             |
| `Pairs logs/`          | Contains execution logs and trading summaries               |
| `Pair_visualizations/` | Plots of z-scores, trades, capital curves for each pair     |

---

## How to Run

### 1. **One-time Setup**

- Clone this repo and set up a Python environment (Python 3.8+ recommended).
- pip install -r requirements.txt

### 2. **Generation of buy and sell signals based on SMA and Rolling-Regression**

- Run the following command to generate **optimized_results.csv** using SMA and Rolling-Regression:
- python smaDDBacktester.py

This script performs the following key functions:

- Downloads OHLCV data for all NASDAQ-100 and S&P 500 tickers using Yahoo Finance
- Automatically merges and updates the latest dataset while cleaning up outdated files
- Backtests strategies using either:
  - Simple Moving Average (SMA) crossover
  - Rolling regression slope crossover
- Optimizes short and long window sizes (SMA_S and SMA_L) using:
  - Performance (return) or
  - F1-Score for predictive quality
- Generates signals (Buy or Sell) based on the latest crossover position
- Saves results to optimized_results.csv and a date-stamped file
- Scheduled run daily at 1:00 AM (New York time) using schedule
- Skips execution on weekends and US holidays using holidays library
- Includes performance metrics: Accuracy, Precision, Recall, F1 Score, Max Drawdown

### 3. **Run the Trading Bot**

- Run the following to execute trades
- python main.py

This script reads the optimized strategy results and places orders using Alpaca’s paper trading API:

- Buys based on buy signals
- Shorts based on sell signals
- Shorts are placed as Limit orders
- Buys and Shorts are executed based on existing positions or pending orders.
- Profitable positions that are >5% are closed before market close
- Tries to maintain positive cash balance
- Automatically skips weekends and holidays

### 4. Plot Signal Comparison — SMA vs Regression

- python plotting.py

This script generates visual comparisons between SMA crossover signals and rolling regression slope signals:

- Uses the optimized SMA short & long windows from optimized_results.csv
- Computes rolling regression slope indicators
- Identifies Buy/Sell signals based on:
  - SMA crossover
  - Slope crossover
- Plots both price and indicator curves along with signal markers
- Saves signal comparison plots to compare_signals_plots/ folder
- Automatically skips tickers with missing data
- Helps visually analyze and compare signal quality

### 5. Plot Confusion Matrices for Slope-Based Predictions

- python confusion_plots.py

This script evaluates and visualizes prediction performance of regression-based signals:

- Uses slope crossover strategy to predict upward/downward price movement
- Compares predictions with actual log return movement (up/down)
- Calculates classification metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- Plots annotated confusion matrices for each symbol
- Saves plots to confusion_plots/ folder
- Useful for debugging and validating signal reliability across tickers

### 6. **Pairs Trading Logic – Overview**

Open and execute the notebook `pairs_trader.ipynb`. It performs the following:

#### Step 1: **Identify Cointegrated Pairs**

- Downloads historical stock data
- Tests for cointegration using statistical tests
- Selects pairs with stable long-term price relationships

#### Step 2: **Backtest Trading Strategy**

- For each cointegrated pair:

  - Computes hedge ratio using linear regression
  - Builds a **spread** and corresponding **z-score**
  - Generates **Buy/Sell signals** based on z-score thresholds:
    - **Buy (long spread)** when z-score < –entry threshold
    - **Sell (short spread)** when z-score > entry threshold
    - **Exit** when z-score reverts toward 0 or hits a stop-loss

- **Backtest results include**:
  - Final capital
  - Total return (%)
  - Win rate
  - Average trade return
  - Sharpe ratio
  - Maximum drawdown
  - Holding periods

#### Step 3: **Logging Results**

- Logs are saved in `Pairs logs/` folder
- Includes trade-level details and summary stats

#### Step 4: **Visualization**

- Plots generated in `Pairs_trading_visualizations/` include:
  - Z-score vs threshold with trade markers
  - Capital curve over time
  - Drawdown visualization

---
