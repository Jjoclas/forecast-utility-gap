# forecast-utility-gap
Bridging the gap between forecasting accuracy and utility: case study with binary options

## Overview
This project explores the relationship between forecasting accuracy and utility in financial markets, with a specific focus on binary options trading. It implements various forecasting models and evaluates their performance not just on traditional accuracy metrics, but also on their utility in real-world trading scenarios.

## Setup Instructions

### Environment Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/forecast-utility-gap.git
cd forecast-utility-gap
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Collection
The project uses financial data from various sources including forex pairs, commodities, cryptocurrencies, and stock indices. To fetch the data:

1. Get your API key from [Polygon.io](https://polygon.io/)
2. Open `notebooks/retrieval.ipynb`
3. Add your API key to the notebook:
```python
API_KEY = "your_polygon_api_key_here"
```
4. Run the notebook to fetch data for all assets. The data will be saved as CSV files in the project directory.

Available assets include:
- Forex pairs (e.g., EURUSD, USDJPY, GBPUSD)
- Commodities (XAUUSD, XAGUSD)
- Cryptocurrencies (BTCUSD, ETHUSD, LTCUSD, XRPUSD)
- Stock indices (AAPL, MSFT, AMZN, GOOGL)

## Technical Indicators

The project uses a comprehensive set of technical indicators for model training, implemented using the TA-Lib library. Here are the main categories of indicators used:

### Overlapping Studies
- Simple Moving Averages (SMA5, SMA10)
- Bollinger Bands (Upper, Middle, Lower)

### Momentum Indicators
- Aroon Oscillator (AARON)
- Average Directional Index (ADX)
- MACD (Moving Average Convergence Divergence)
- Relative Strength Index (RSI)
- Rate of Change (ROC)
- Money Flow Index (MFI)
- Directional Movement Index (DX)
- Directional Indicators (PLUS_DI, MINUS_DI)
- Directional Movement (PLUS_DM, MINUS_DM)
- Stochastic (STOCH, STOCHF, STOCHRSI)
- TRIX
- Ultimate Oscillator (ULTOSC)
- Williams %R (WILLR)

### Volatility Indicators
- Average True Range (ATR)
- Normalized ATR (NATR)
- True Range (TRANGE)

### Volume Indicators
- On Balance Volume (OBV)
- Accumulation/Distribution Line (AD)

### Cycle Indicators
- Hilbert Transform Trendline (HT_TRENDLINE)

### Statistical Functions
- Time Series Forecast (TSF)
- Variance (VAR)

### Price Transform
- Exponential Moving Averages (EMA5, EMA10, EMA50, EMA100, EMA200)

### Candlestick Patterns
- Three Inside Up/Down (CDL3INSIDE)
- Three Outside Up/Down (CDL3OUTSIDE)
- Advance Block (CDLADVANCEBLOCK)
- Belt-hold (CDLBELTHOLD)
- Doji Patterns (CDLDOJI, CDLDOJISTAR, CDLDRAGONFLYDOJI)
- Engulfing Pattern (CDLENGULFING)
- Evening Star (CDLEVENINGSTAR)
- Hammer (CDLHAMMER)
- Harami Pattern (CDLHARAMI)
- And many more candlestick patterns...

All indicators are properly normalized and standardized to ensure consistent scale across different price ranges and time periods.

## Model Training

To train the models, run:
```bash
python run_pipeline.py
```

This will:
1. Process the raw data from the CSV files
2. Train multiple forecasting models
3. Generate performance metrics
4. Save the results in the `metrics` folder as CSV files

The metrics files (e.g., `metrics_neg_log_loss_0.9.csv`, `metrics_neg_log_loss_0.75.csv`, etc.) contain detailed performance evaluations for different model configurations and thresholds.

## Project Structure
```
├── model_pipeline/     # Core model training pipeline
├── metrics/           # Performance metrics and evaluation results
├── utils/            # Utility functions and configurations
├── notebooks/        # Jupyter notebooks for data analysis and visualization
└── candles/         # Candlestick pattern analysis
```




