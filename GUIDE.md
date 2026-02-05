# Investment Score Aggregator

Comprehensive multi-factor stock evaluation system for Google Colab.

## Purpose

Analyzes stocks across 10 quantitative dimensions to identify optimal investment candidates for options trading strategies.

## Metrics Evaluated

### 1. Memory Index (252-day window)
- Measures long-term pattern persistence
- Uses multi-scale variance analysis
- Higher values indicate trending behavior
- Lower values suggest mean-reversion

### 2. Volatility-Adjusted Performance
- Return efficiency normalized by risk
- Annualized excess returns divided by volatility
- Higher values indicate better risk-adjusted returns

### 3-5. Weighted Trend Estimators
- **Monthly (21 days)**: Medium-term momentum
- **Weekly (5 days)**: Short-term momentum  
- **Daily (1 day)**: Immediate direction
- Uses recency-weighted regression
- Recent prices have exponentially higher influence

### 6. Adaptive Forecast (90-day training, 5-day projection)
- Multi-level exponential smoothing
- Projects expected 5-day return percentage
- Learns level, trend, and seasonal patterns

### 7. Velocity Indicator (21-day)
- Rate of change measurement
- Simple momentum metric
- Percentage change from 21 trading days ago

### 8. Dual Average Histogram
- Exponential moving average convergence system
- Histogram shows momentum strength/direction
- Positive values suggest bullish momentum

### 9. Momentum Oscillator
- Bounded 0-100 indicator
- Based on gain/loss ratios
- **Target value: 46** (neither overbought/oversold)
- **Warning value: 78** (overbought condition)

### 10. Cumulative Volume Flow
- Tracks volume accumulation vs distribution
- Positive when price rises with volume
- Negative when price falls with volume

## Google Colab Usage

### Prerequisites

```python
# Install required packages
!pip install yfinance pandas numpy
```

### File Setup

1. **Input file location:**
   ```
   /content/drive/MyDrive/Outputs/t/Test_ticker.txt
   ```
   
2. **Input file format** (one symbol per line):
   ```
   AAPL
   MSFT
   GOOGL
   ```

3. **Output file location:**
   ```
   /content/drive/MyDrive/Outputs/Bullish9_Feb5.csv
   ```

### Execution

```python
# Run the analysis
!python /content/drive/MyDrive/investment_scorer.py
```

## Scoring Methodology

### Standard Metrics (9 of 10)
- Raw values normalized to 0-10 scale
- **10**: Best performer in group
- **0**: Worst performer in group
- Linear interpolation between extremes

### Oscillator Special Scoring (1 of 10)
- **Value 46 → Score 10** (optimal target)
- **Value 78 → Score 0** (extreme warning)
- Score decreases linearly with distance from 46

### Composite Score
- Sum of all 10 individual scores
- **Range**: 0 to 100 points
- Higher scores indicate stronger multi-factor performance

## Output Structure

CSV file columns:

```
Symbol                  - Ticker code
LatestClose            - Current price
CompositeScore         - Total of all scores (0-100)
MemoryIndex            - Raw memory index value
Score_MemoryIndex      - Normalized score (0-10)
VolatilityPerformance  - Raw vol-adjusted return
Score_VolPerformance   - Normalized score (0-10)
[... similar pattern for all 10 metrics ...]
```

## Investment Context

### Target Company Profile
- Market capitalization > $2.5B
- Daily trading volume > $1.5B
- TTM EPS: Positive
- High memory index (trending)
- High volatility-adjusted performance

### Recommended Options Strategy
- **Type**: Long call options
- **Expiration**: 45-60 days
- **Delta range**: 0.3 to 0.4
- **Goal**: Regular income through directional trades

### Risk Disclosure
- Rankings based on historical data only
- Past performance ≠ future results
- Combine with fundamental analysis
- Use proper position sizing
- Not financial advice

## Technical Specifications

### Dependencies
```
yfinance >= 0.2.0
pandas >= 1.3.0
numpy >= 1.21.0
```

### Data Requirements
- Internet connectivity (yfinance API)
- Minimum 40 trading days of history per ticker
- Free tier sufficient for typical usage

### Performance
- Processing speed: ~1-2 tickers/second
- Memory scales with ticker count
- Google Colab free tier adequate

## Configuration Options

### Adjustable Parameters

Edit within source code:

```python
# Pattern memory lookback
memory_coefficient = pattern_memory.calculate(maximum_window=252)

# Trend analysis windows
trend_m = trend_monthly.calculate(observation_count=21)  # Monthly
trend_w = trend_weekly.calculate(observation_count=5)    # Weekly
trend_d = trend_daily.calculate(observation_count=1)     # Daily

# Forecast parameters
forecast_engine.calculate(calibration_length=90, steps_ahead=5)

# Oscillator targets
normalizer.normalize_oscillator(values, target_optimal=46, target_extreme=78)
```

### Smoothing Coefficients

In `AdaptiveForecastEngine.calculate()`:

```python
learning_level = 0.38    # Level adaptation rate
learning_trend = 0.14    # Trend adaptation rate
learning_season = 0.09   # Seasonal adaptation rate
```

### Weighting Factors

In `WeightedTrendEstimator.calculate()`:

```python
recency_factor = 0.18    # Controls exponential weighting
```

## Troubleshooting

### "Cannot locate input file"
- Verify Google Drive is mounted
- Check path matches exactly: `/content/drive/MyDrive/Outputs/t/Test_ticker.txt`
- Ensure file exists with correct name

### "No successful computations"
- Verify internet connection
- Check ticker symbols are valid
- Some tickers may fail (insufficient data)
- Try with fewer tickers first

### Rate Limiting
- yfinance has API rate limits
- Script processes sequentially to mitigate
- Add delays between batches if needed

## Example Output

```
TOP 10 INVESTMENT CANDIDATES
Symbol  LatestClose  CompositeScore  MemoryIndex  VolatilityPerformance  ForecastReturnPct  OscillatorReading
NVDA    445.30       89.2            0.66         2.52                   3.45              47.8
META    325.15       86.7            0.63         2.38                   3.10              46.2
AAPL    178.25       84.1            0.61         2.15                   2.75              48.5
```

## Directory Structure

```
/content/drive/MyDrive/Outputs/
├── t/
│   └── Test_ticker.txt          # Input: ticker list
├── Bullish9_Feb5.csv            # Output: scored results
└── investment_scorer.py         # Executable script
```

## License

For educational and personal research only.
