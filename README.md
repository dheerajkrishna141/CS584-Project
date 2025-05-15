# Stock Market Data Analysis and Prediction: Using Technical Strategies

This project is part of a CS course on the **Theory and Applications of Data Mining**, focusing on implementing and optimizing stock trading strategies using historical stock data and prediction based on modeling techniques learned from the class.

The core functionality involves backtesting and optimizing **Pairs Trading** strategy.

---

### Project Structure

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

## 2. **Pairs Trading Logic – Overview**

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
