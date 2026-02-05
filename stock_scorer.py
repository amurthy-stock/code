#!/usr/bin/env python3
"""
Investment Stock Scoring System
Custom implementation for multi-factor analysis
"""

import yfinance as yf
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Colab integration
try:
    from google.colab import drive
    drive.mount('/content/drive')
except:
    pass

@dataclass
class PathConfig:
    """File path configuration"""
    input_path: str = '/content/drive/MyDrive/Outputs/t/Test_ticker.txt'
    output_path: str = '/content/drive/MyDrive/Outputs/Bullish9_Feb5.csv'


def read_symbols_file(filepath: str) -> List[str]:
    """Load stock symbols from text file"""
    with open(filepath, 'r') as f:
        return [s.strip() for s in f if s.strip()]


def fetch_stock_data(symbol: str, days: int = 504) -> Tuple[np.ndarray, np.ndarray]:
    """Get historical price and volume data"""
    print(f"Fetching {symbol}...")
    df = yf.download(symbol, period='2y', interval='1d', progress=False, auto_adjust=True)
    if df.empty or len(df) < 252:
        return None, None
    return df['Close'].values, df['Volume'].values


def detrended_fluctuation_hurst(price_seq: np.ndarray, trading_days: int = 252) -> float:
    """
    Calculate persistence using Detrended Fluctuation Analysis
    Alternative to R/S Hurst - measures long-term memory
    """
    if len(price_seq) < trading_days:
        return np.nan
    
    subset = price_seq[-trading_days:]
    log_prices = np.log(subset)
    deviations = np.diff(log_prices)
    cumulative = np.cumsum(deviations - np.mean(deviations))
    
    # Multiple window sizes
    windows = [8, 16, 32, 64, 128]
    fluctuations = []
    
    for win in windows:
        if win * 4 > len(cumulative):
            continue
        
        segments = len(cumulative) // win
        local_flucts = []
        
        for seg in range(segments):
            segment_data = cumulative[seg*win:(seg+1)*win]
            trend_line = np.polyfit(range(win), segment_data, 1)
            trend_vals = np.polyval(trend_line, range(win))
            local_flucts.append(np.sqrt(np.mean((segment_data - trend_vals)**2)))
        
        if local_flucts:
            fluctuations.append(np.mean(local_flucts))
    
    if len(fluctuations) < 3:
        return np.nan
    
    # Power law fit
    valid_windows = windows[:len(fluctuations)]
    log_fit = np.polyfit(np.log(valid_windows), np.log(fluctuations), 1)
    persistence_exp = log_fit[0]
    
    return float(persistence_exp)


def reward_volatility_ratio(price_seq: np.ndarray, benchmark_rate: float = 0.02) -> float:
    """
    Calculate reward-to-risk metric using geometric returns
    Alternative approach to standard Sharpe calculation
    """
    if len(price_seq) < 30:
        return np.nan
    
    # Geometric returns
    log_returns = np.diff(np.log(price_seq))
    
    if len(log_returns) == 0:
        return np.nan
    
    # Annualized geometric mean
    geom_mean = np.exp(np.mean(log_returns) * 252) - 1
    
    # Downside deviation (semi-variance approach)
    negative_rets = log_returns[log_returns < 0]
    if len(negative_rets) > 0:
        downside_vol = np.std(negative_rets) * np.sqrt(252)
    else:
        downside_vol = np.std(log_returns) * np.sqrt(252)
    
    if downside_vol == 0:
        return np.nan
    
    ratio = (geom_mean - benchmark_rate) / downside_vol
    return float(ratio)


def polynomial_direction(price_seq: np.ndarray, lookback: int) -> float:
    """
    Quadratic trend strength measurement
    Uses 2nd order polynomial for better curve fitting
    """
    if len(price_seq) < lookback:
        return np.nan
    
    recent = price_seq[-lookback:]
    indices = np.arange(len(recent))
    
    # Quadratic fit
    poly_coeffs = np.polyfit(indices, recent, 2)
    
    # Evaluate at endpoints
    start_fit = np.polyval(poly_coeffs, 0)
    end_fit = np.polyval(poly_coeffs, len(recent) - 1)
    
    # Percentage trajectory
    trajectory = ((end_fit - start_fit) / start_fit) * 100 if start_fit != 0 else 0
    
    return float(trajectory)


def exponential_projection(price_seq: np.ndarray, history_window: int, future_steps: int) -> Tuple[float, float]:
    """
    Exponential weighted projection instead of linear
    Gives more weight to recent data points
    """
    if len(price_seq) < history_window:
        return np.nan, np.nan
    
    recent = price_seq[-history_window:]
    
    # Exponential weights (more recent = higher weight)
    weights = np.exp(np.linspace(-2, 0, len(recent)))
    weights = weights / np.sum(weights)
    
    # Weighted polynomial fit
    indices = np.arange(len(recent))
    fit_coeffs = np.polyfit(indices, recent, 1, w=weights)
    
    # Project forward
    future_idx = len(recent) + future_steps - 1
    projected_val = np.polyval(fit_coeffs, future_idx)
    
    curr_price = price_seq[-1]
    percent_gain = ((projected_val - curr_price) / curr_price) * 100
    
    return float(projected_val), float(percent_gain)


def velocity_indicator(price_seq: np.ndarray, span: int = 21) -> float:
    """
    Price velocity measurement over time period
    Custom momentum calculation
    """
    if len(price_seq) <= span:
        return np.nan
    
    current = price_seq[-1]
    historical = price_seq[-(span + 1)]
    
    velocity = ((current - historical) / historical) * 100
    return float(velocity)


def dual_ema_divergence(price_seq: np.ndarray) -> float:
    """
    Calculate dual exponential moving average spread
    Custom convergence-divergence indicator
    """
    if len(price_seq) < 40:
        return np.nan
    
    series = pd.Series(price_seq)
    
    # Two different decay rates
    fast_decay = series.ewm(alpha=0.15, adjust=False).mean()
    slow_decay = series.ewm(alpha=0.075, adjust=False).mean()
    
    spread = fast_decay - slow_decay
    signal_smooth = spread.ewm(alpha=0.2, adjust=False).mean()
    
    divergence_metric = spread.iloc[-1] - signal_smooth.iloc[-1]
    
    return float(divergence_metric)


def strength_index_oscillator(price_seq: np.ndarray, span: int = 14) -> float:
    """
    Custom momentum oscillator based on directional strength
    Alternative to standard RSI formula
    """
    if len(price_seq) <= span:
        return np.nan
    
    changes = np.diff(price_seq)
    
    # Separate up and down movements
    up_moves = np.abs(changes[changes > 0])
    down_moves = np.abs(changes[changes < 0])
    
    # Exponential smoothing of movements
    up_series = pd.Series(up_moves if len(up_moves) > 0 else [0])
    down_series = pd.Series(down_moves if len(down_moves) > 0 else [0])
    
    avg_up = up_series.ewm(span=span, adjust=False).mean().iloc[-1]
    avg_down = down_series.ewm(span=span, adjust=False).mean().iloc[-1]
    
    if avg_down == 0:
        return 100.0
    
    strength_ratio = avg_up / avg_down
    oscillator = 100.0 - (100.0 / (1.0 + strength_ratio))
    
    return float(oscillator)


def directional_volume_flow(price_seq: np.ndarray, volume_seq: np.ndarray) -> float:
    """
    Cumulative directional volume indicator
    Tracks institutional flow patterns
    """
    if volume_seq is None or len(price_seq) < 2:
        return np.nan
    
    flow_accumulator = 0.0
    
    for idx in range(1, len(price_seq)):
        price_change = price_seq[idx] - price_seq[idx - 1]
        
        if price_change > 0:
            flow_accumulator += volume_seq[idx]
        elif price_change < 0:
            flow_accumulator -= volume_seq[idx]
    
    return float(flow_accumulator)


def map_to_decile(val: float, min_bound: float, max_bound: float, reverse: bool = False) -> float:
    """Transform value to 0-10 scale with bounds checking"""
    if np.isnan(val) or np.isnan(min_bound) or np.isnan(max_bound):
        return 0.0
    
    if max_bound == min_bound:
        return 5.0
    
    normalized = (val - min_bound) / (max_bound - min_bound)
    
    if reverse:
        normalized = 1.0 - normalized
    
    decile = normalized * 10.0
    return max(0.0, min(10.0, decile))


def custom_oscillator_scoring(osc_val: float, ideal: float = 46.0, poor: float = 78.0) -> float:
    """
    Non-linear scoring for oscillator with target value
    Penalizes deviation from ideal more harshly
    """
    if np.isnan(osc_val):
        return 0.0
    
    if osc_val <= ideal:
        # Below target: quadratic reward
        ratio = osc_val / ideal
        score = 10.0 * (ratio ** 0.5)
    else:
        # Above target: exponential penalty
        excess = (osc_val - ideal) / (poor - ideal)
        score = 10.0 * np.exp(-2 * excess)
    
    return max(0.0, min(10.0, score))


def process_single_stock(symbol: str) -> dict:
    """Analyze one stock and return metrics dictionary"""
    prices, volumes = fetch_stock_data(symbol)
    
    if prices is None:
        return None
    
    # Extract data windows
    full_year = prices[-252:] if len(prices) >= 252 else prices
    quarter = prices[-90:] if len(prices) >= 90 else prices
    month = prices[-21:] if len(prices) >= 21 else prices
    week = prices[-5:] if len(prices) >= 5 else prices
    
    full_year_vol = volumes[-252:] if volumes is not None and len(volumes) >= 252 else None
    
    return {
        'Symbol': symbol,
        'CurrentPrice': float(prices[-1]),
        'PersistenceExp': detrended_fluctuation_hurst(prices, 252),
        'RewardRisk': reward_volatility_ratio(full_year),
        'MonthlyDirection': polynomial_direction(month, len(month)) if len(month) >= 5 else np.nan,
        'WeeklyDirection': polynomial_direction(week, len(week)) if len(week) >= 3 else np.nan,
        'DailyDirection': polynomial_direction(full_year, len(full_year)) if len(full_year) >= 20 else np.nan,
        'ProjectedPrice': exponential_projection(quarter, len(quarter), 5)[0] if len(quarter) >= 20 else np.nan,
        'ProjectedGain': exponential_projection(quarter, len(quarter), 5)[1] if len(quarter) >= 20 else np.nan,
        'VelocityMetric': velocity_indicator(full_year, 21),
        'DualEmaDivergence': dual_ema_divergence(full_year),
        'StrengthOscillator': strength_index_oscillator(full_year, 14),
        'VolumeFlow': directional_volume_flow(full_year, full_year_vol) if full_year_vol is not None else np.nan
    }


def apply_scoring_system(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Transform raw metrics to scored values"""
    
    # Standard metrics (higher is better)
    standard_cols = [
        ('PersistenceExp', 'PersistScore'),
        ('RewardRisk', 'RewardScore'),
        ('MonthlyDirection', 'MonthScore'),
        ('WeeklyDirection', 'WeekScore'),
        ('DailyDirection', 'DayScore'),
        ('ProjectedGain', 'ProjectScore'),
        ('VelocityMetric', 'VelocityScore'),
        ('DualEmaDivergence', 'DivergenceScore'),
        ('VolumeFlow', 'FlowScore')
    ]
    
    for raw_col, score_col in standard_cols:
        min_v = dataframe[raw_col].min()
        max_v = dataframe[raw_col].max()
        dataframe[score_col] = dataframe[raw_col].apply(
            lambda x: map_to_decile(x, min_v, max_v, False)
        )
    
    # Special oscillator scoring
    dataframe['OscillatorScore'] = dataframe['StrengthOscillator'].apply(
        custom_oscillator_scoring
    )
    
    # Aggregate scoring
    score_cols = [col for col in dataframe.columns if 'Score' in col]
    dataframe['AggregateScore'] = dataframe[score_cols].sum(axis=1)
    
    return dataframe


def main_execution():
    """Primary execution flow"""
    config = PathConfig()
    
    print("=" * 70)
    print("STOCK SCORING SYSTEM - Multi-Factor Investment Analysis")
    print("=" * 70)
    
    symbols = read_symbols_file(config.input_path)
    print(f"\nProcessing {len(symbols)} symbols...")
    
    results_list = []
    for sym in symbols:
        result = process_single_stock(sym)
        if result:
            results_list.append(result)
    
    if not results_list:
        print("No valid results generated")
        return
    
    print(f"\nSuccessfully analyzed {len(results_list)} stocks")
    
    # Build dataframe
    df = pd.DataFrame(results_list)
    
    # Apply scoring
    df = apply_scoring_system(df)
    
    # Sort by aggregate score
    df = df.sort_values('AggregateScore', ascending=False)
    
    # Column organization
    output_cols = [
        'Symbol', 'CurrentPrice',
        'PersistenceExp', 'PersistScore',
        'RewardRisk', 'RewardScore',
        'MonthlyDirection', 'MonthScore',
        'WeeklyDirection', 'WeekScore',
        'DailyDirection', 'DayScore',
        'ProjectedPrice', 'ProjectedGain', 'ProjectScore',
        'VelocityMetric', 'VelocityScore',
        'DualEmaDivergence', 'DivergenceScore',
        'StrengthOscillator', 'OscillatorScore',
        'VolumeFlow', 'FlowScore',
        'AggregateScore'
    ]
    
    final_df = df[output_cols]
    
    # Save output
    try:
        final_df.to_csv(config.output_path, index=False)
        print(f"\n✓ Saved to: {config.output_path}")
    except:
        alt_path = '/content/Bullish9_Feb5.csv'
        final_df.to_csv(alt_path, index=False)
        print(f"\n✓ Saved to: {alt_path}")
    
    # Display top results
    print("\n" + "=" * 70)
    print("TOP 10 INVESTMENT CANDIDATES")
    print("=" * 70)
    display_cols = ['Symbol', 'CurrentPrice', 'AggregateScore', 'PersistenceExp', 'RewardRisk']
    print(final_df[display_cols].head(10).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main_execution()
