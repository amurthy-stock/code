# Stock Analysis Tools

This repository contains two different stock analysis tools for Google Colab:

## 1. Stock Scorer (`stock_scorer.py`) - ⭐ NEW

**Purpose**: Multi-factor investment scoring system for options trading

**Features**:
- Detrended Fluctuation Analysis for persistence measurement
- Risk-adjusted returns with downside deviation
- Multi-timeframe trend analysis (monthly, weekly, daily)
- Exponentially weighted 5-day price projection
- Comprehensive technical indicators (velocity, EMA divergence, strength oscillator, volume flow)
- Normalized 0-10 scoring system for all metrics
- Aggregate scoring for investment ranking

**Input**: `/content/drive/MyDrive/Outputs/t/Test_ticker.txt`
**Output**: `/content/drive/MyDrive/Outputs/Bullish9_Feb5.csv`

**Documentation**: See [GUIDE.md](GUIDE.md) for detailed usage

**Usage**:
```python
!python stock_scorer.py
```

**Target**: Options traders looking for long call opportunities (45-60 DTE, 0.3-0.4 Delta)

---

## 2. Fibonacci LSTM Forecast (`success6.txt`)

**Purpose**: Multi-timeframe LSTM neural network forecasting

**Features**:
- Weekly, daily, and hourly LSTM models
- Fibonacci retracement point analysis
- Deep learning predictions with early stopping
- Visualization with cumulative gain charts
- MAPE (Mean Absolute Percentage Error) metrics

**Input**: `/content/drive/MyDrive/Outputs/Test_ticker.txt`
**Output**: `/content/drive/MyDrive/Outputs/Fibonacci_LSTM_results.csv`

**Usage**: Copy contents of `success6.txt` into Google Colab and run

**Note**: Requires TensorFlow and may take longer to execute due to model training

---

## Choosing Between Tools

| Feature | stock_scorer.py | success6.txt |
|---------|-----------------|--------------|
| Approach | Statistical/Technical | Deep Learning |
| Speed | Fast (seconds per stock) | Slower (model training) |
| Scoring | Yes (0-10 scale) | No (raw predictions) |
| Dependencies | yfinance, pandas, numpy | + TensorFlow, sklearn |
| Complexity | Moderate | High |
| Best For | Quick screening | Detailed forecasts |

---

## Installation

Both scripts work in Google Colab without additional installation.

For local use:
```bash
# For stock_scorer.py
pip install yfinance pandas numpy

# For success6.txt (additional)
pip install tensorflow scikit-learn matplotlib tqdm
```

---

## Input File Format

Create a text file with one stock ticker per line:
```
AAPL
MSFT
GOOGL
TSLA
NVDA
```

---

## Testing

Run validation tests:
```bash
python validate_scorer.py
```

---

## Investment Strategy

Both tools support identifying stocks for:
- Market cap > $2.5B
- Volume > 1.5B shares  
- Positive TTM EPS
- Long call options strategy

The `stock_scorer.py` provides more comprehensive scoring for decision-making, while `success6.txt` offers deep learning forecasts.

---

## License

MIT License

---

## Support

For issues or questions, please open an issue in the repository.
