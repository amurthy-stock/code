# Stock Scoring System

Multi-factor investment analysis tool for Google Colab with custom technical indicators and comprehensive scoring.

## Quick Start

### Google Colab Usage

1. Create input file with stock symbols (one per line):
   ```
   /content/drive/MyDrive/Outputs/t/Test_ticker.txt
   ```

2. Run the script:
   ```python
   !python stock_scorer.py
   ```

3. Results saved to:
   ```
   /content/drive/MyDrive/Outputs/Bullish9_Feb5.csv
   ```

## Investment Philosophy

Designed for options traders targeting:
- Long call positions (45-60 DTE, 0.3-0.4 Delta)
- Companies with Market Cap > $2.5B
- Trading Volume > 1.5B shares
- Positive TTM EPS

## Technical Indicators

### 1. Persistence Exponent (Hurst-like)
- **Method**: Detrended Fluctuation Analysis (DFA)
- **Window**: 252 trading days
- **Interpretation**: 
  - > 0.5 = Trending behavior
  - < 0.5 = Mean-reverting behavior
- **Unique Approach**: Uses fluctuation analysis instead of traditional R/S method

### 2. Reward-Risk Ratio (Sharpe-like)
- **Method**: Geometric returns with downside deviation
- **Calculation**: Uses semi-variance for risk measurement
- **Benefit**: Penalizes downside volatility more than upside
- **Annualized**: Yes (252 trading days)

### 3. Direction Indicators
Three timeframes analyzed:
- **Monthly** (21-day window)
- **Weekly** (5-day window)  
- **Daily** (252-day window)

**Method**: Quadratic polynomial fitting for better curve detection

### 4. Price Projection
- **History Window**: 90 days
- **Forecast**: 5 days ahead
- **Method**: Exponentially weighted regression (recent data weighted higher)
- **Output**: Projected price + expected gain %

### 5. Velocity Metric
- **Period**: 21 days
- **Calculation**: Rate of price change over time
- **Use**: Momentum measurement

### 6. Dual EMA Divergence
- **Fast EMA**: Alpha = 0.15
- **Slow EMA**: Alpha = 0.075
- **Signal**: Smoothed with alpha = 0.2
- **Output**: Histogram value (spread minus signal)

### 7. Strength Oscillator
- **Period**: 14 days
- **Range**: 0-100
- **Method**: Exponentially smoothed up/down movements
- **Scoring**: 
  - 46 = Optimal (score 10)
  - 78 = Overbought/worst (score 0)
  - Non-linear penalty function

### 8. Volume Flow
- **Method**: Cumulative directional volume
- **Tracks**: Institutional buying/selling pressure
- **Interpretation**: Positive = accumulation, Negative = distribution

## Scoring System

Each metric converted to 0-10 scale:
- **0** = Worst performer in group
- **10** = Best performer in group
- **5** = Middle of range

### Aggregate Score
Sum of all individual scores (max 100 points):
- Persistence Score (0-10)
- Reward Score (0-10)
- Month Score (0-10)
- Week Score (0-10)
- Day Score (0-10)
- Project Score (0-10)
- Velocity Score (0-10)
- Divergence Score (0-10)
- Oscillator Score (0-10)
- Flow Score (0-10)

Results sorted by Aggregate Score (highest to lowest).

## Output Format

CSV file with columns:

| Column | Description |
|--------|-------------|
| Symbol | Stock ticker |
| CurrentPrice | Latest close price |
| PersistenceExp | DFA Hurst value |
| PersistScore | Normalized 0-10 |
| RewardRisk | Risk-adjusted return |
| RewardScore | Normalized 0-10 |
| MonthlyDirection | 21-day trend % |
| MonthScore | Normalized 0-10 |
| WeeklyDirection | 5-day trend % |
| WeekScore | Normalized 0-10 |
| DailyDirection | 252-day trend % |
| DayScore | Normalized 0-10 |
| ProjectedPrice | 5-day forecast |
| ProjectedGain | Expected return % |
| ProjectScore | Normalized 0-10 |
| VelocityMetric | 21-day momentum |
| VelocityScore | Normalized 0-10 |
| DualEmaDivergence | EMA histogram |
| DivergenceScore | Normalized 0-10 |
| StrengthOscillator | 0-100 oscillator |
| OscillatorScore | Special 0-10 scoring |
| VolumeFlow | Cumulative volume |
| FlowScore | Normalized 0-10 |
| AggregateScore | Total score (sum) |

## Configuration

Edit paths at top of `stock_scorer.py`:

```python
@dataclass
class PathConfig:
    input_path: str = '/content/drive/MyDrive/Outputs/t/Test_ticker.txt'
    output_path: str = '/content/drive/MyDrive/Outputs/Bullish9_Feb5.csv'
```

## Requirements

```
yfinance
numpy
pandas
```

Pre-installed in Google Colab. For local use:
```bash
pip install yfinance numpy pandas
```

## Algorithm Uniqueness

This implementation uses several unique approaches:

1. **DFA Method**: Replaces traditional Hurst R/S analysis with detrended fluctuation analysis
2. **Downside Risk**: Uses semi-variance instead of standard deviation for risk
3. **Quadratic Trends**: Uses 2nd-order polynomials instead of linear regression
4. **Exponential Weighting**: Recent prices weighted more heavily in projections
5. **Custom Oscillator**: Non-linear scoring function with exponential penalties
6. **Dual EMA**: Custom alpha values optimized for stock analysis

## Data Requirements

- Minimum 252 days of history required
- Daily interval data
- Handles missing/insufficient data gracefully

## Limitations

- Requires network access for Yahoo Finance data
- Historical data only (not real-time)
- Linear projections may not capture regime changes
- Past performance ≠ future results

## License

MIT License
