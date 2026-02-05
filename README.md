# Stock Analysis for Investment Decision Making

Custom-built quantitative analysis system for Google Colab environments.

## Quick Start

**For Google Colab:**

1. Upload `investment_scorer.py` to your Google Drive
2. Create input file at: `/content/drive/MyDrive/Outputs/t/Test_ticker.txt`
3. Add ticker symbols (one per line)
4. Run: `!python /content/drive/MyDrive/investment_scorer.py`
5. Results saved to: `/content/drive/MyDrive/Outputs/Bullish9_Feb5.csv`

## Documentation

See [GUIDE.md](GUIDE.md) for complete documentation including:
- Detailed metric descriptions
- Configuration options
- Troubleshooting guide
- Example outputs

## What It Does

Evaluates stocks across 10 quantitative factors:
- Pattern memory analysis (252-day)
- Volatility-adjusted performance
- Multi-timeframe trends (monthly, weekly, daily)
- Adaptive 5-day forecasts
- Momentum indicators
- Volume flow analysis

Each metric scored 0-10, combined into composite score (0-100).

## Investment Strategy

Designed for options traders seeking:
- Long call positions
- 45-60 DTE
- 0.3-0.4 Delta
- $2.5B+ market cap companies
- Positive earnings

## Requirements

```
yfinance
pandas
numpy
```

## Note

For educational and research purposes. Not financial advice.
