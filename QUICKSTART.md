# Quick Reference Card

## Stock Scorer - Fast Start Guide

### Setup (One Time)
1. Create folder in Google Drive: `MyDrive/Outputs/t/`
2. Create ticker file: `Test_ticker.txt`
3. Add stock symbols (one per line)

### Run in Google Colab
```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Run scorer
!python stock_scorer.py
```

### Output Location
`/content/drive/MyDrive/Outputs/Bullish9_Feb5.csv`

---

## Understanding Your Results

### Key Columns

| Column | What It Means | Good Value |
|--------|---------------|------------|
| **AggregateScore** | Overall investment score | > 70 |
| **PersistenceExp** | Trend persistence | > 0.5 |
| **RewardRisk** | Risk-adjusted returns | > 1.0 |
| **StrengthOscillator** | Momentum (0-100) | ~46 |
| **ProjectedGain** | 5-day forecast % | Positive |

### Score Meanings
- **0-3**: Weak candidate
- **4-6**: Average candidate  
- **7-8**: Good candidate
- **9-10**: Excellent candidate

### Top Performer Indicators
Look for stocks with:
- ✓ AggregateScore > 70
- ✓ Multiple scores 8+
- ✓ Positive ProjectedGain
- ✓ StrengthOscillator near 46

---

## Investment Decision Framework

### Step 1: Screen
- Run stock_scorer.py on your watchlist
- Sort by AggregateScore

### Step 2: Filter
- Keep only AggregateScore > 70
- Check ProjectedGain is positive
- Verify company fundamentals separately:
  - Market Cap > $2.5B
  - Volume > 1.5B
  - Positive TTM EPS

### Step 3: Options Strategy
For qualified stocks:
- **Type**: Long Call
- **DTE**: 45-60 days
- **Delta**: 0.3 to 0.4
- **Entry**: When multiple timeframes align

### Step 4: Monitor
- Re-run weekly
- Watch for score changes
- Adjust positions accordingly

---

## Metric Interpretations

### Persistence Exponent (PersistenceExp)
- **> 0.5**: Trending (momentum continues)
- **< 0.5**: Mean-reverting (bounces back)
- **~0.5**: Random walk

### Reward-Risk Ratio (RewardRisk)
- **> 2.0**: Exceptional risk-adjusted returns
- **1.0-2.0**: Good risk-adjusted returns
- **< 1.0**: Poor risk-adjusted returns

### Direction Scores
- **8-10**: Strong uptrend
- **4-7**: Moderate trend
- **0-3**: Weak or downtrend

### Strength Oscillator
- **< 30**: Oversold (potential bounce)
- **30-50**: Healthy range
- **50-70**: Elevated but acceptable
- **> 70**: Overbought (caution)

### Volume Flow
- **Large Positive**: Strong accumulation
- **Near Zero**: Neutral
- **Large Negative**: Distribution

---

## Common Issues

### "No valid results"
- Check ticker symbols are correct
- Ensure stocks have 252+ days history
- Verify internet connection

### "File not found"
- Double-check path: `/content/drive/MyDrive/Outputs/t/Test_ticker.txt`
- Ensure Drive is mounted
- Check file permissions

### "Insufficient data"
- Stock needs 252+ trading days
- Try different ticker
- Check if recently IPO'd

---

## Tips for Better Results

1. **Diversify Input**: Include 20-50 stocks in your screening list
2. **Regular Updates**: Run weekly to catch changing conditions  
3. **Cross-Verify**: Use fundamental analysis alongside technical scores
4. **Risk Management**: Never put all capital in highest scorer
5. **Track Record**: Keep history to see which scores predict best

---

## Advanced Usage

### Custom Paths
Edit `stock_scorer.py`:
```python
@dataclass
class PathConfig:
    input_path: str = 'YOUR_INPUT_PATH'
    output_path: str = 'YOUR_OUTPUT_PATH'
```

### Batch Analysis
Create multiple input files:
- `Growth_stocks.txt`
- `Value_stocks.txt`
- `Tech_stocks.txt`

Run separately and compare results.

### Integration
Load results into:
- Excel/Google Sheets for further analysis
- Python pandas for custom filtering
- Visualization tools for charting

---

## Need Help?

- **Full Documentation**: See [GUIDE.md](GUIDE.md)
- **Code Examples**: See [colab_example.py](colab_example.py)
- **Tests**: Run `python validate_scorer.py`
- **Issues**: Open GitHub issue

---

## Remember

⚠️ **Disclaimer**: This is a technical analysis tool. It does NOT:
- Guarantee profits
- Replace fundamental analysis
- Account for news/events
- Predict black swan events

Always do your own research and manage risk appropriately.
