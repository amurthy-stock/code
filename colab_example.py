"""
Google Colab Example - Stock Scorer

Copy this entire cell into Google Colab to run the stock analysis
"""

# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Prepare your input file
# Create a file at: /content/drive/MyDrive/Outputs/t/Test_ticker.txt
# with one ticker per line, for example:
"""
AAPL
MSFT
GOOGL
TSLA
NVDA
AMD
META
AMZN
NFLX
"""

# Step 3: Install required packages (if needed)
# Usually pre-installed in Colab
# !pip install yfinance pandas numpy

# Step 4: Clone the repository and run the stock scorer
!git clone https://github.com/amurthy-stock/code.git
%cd code
!python stock_scorer.py

# Step 5: View results
# Results will be saved to: /content/drive/MyDrive/Outputs/Bullish9_Feb5.csv

# Optional: Load and display results in Colab
import pandas as pd

results = pd.read_csv('/content/drive/MyDrive/Outputs/Bullish9_Feb5.csv')

print("Top 10 Investment Candidates:")
print(results[['Symbol', 'CurrentPrice', 'AggregateScore', 'PersistenceExp', 'RewardRisk']].head(10))

# Optional: Visualize top stocks
import matplotlib.pyplot as plt

top_10 = results.head(10)
plt.figure(figsize=(12, 6))
plt.barh(top_10['Symbol'], top_10['AggregateScore'])
plt.xlabel('Aggregate Score')
plt.title('Top 10 Stocks by Aggregate Score')
plt.tight_layout()
plt.show()
