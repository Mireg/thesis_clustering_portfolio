import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import yfinance as yf
from datetime import datetime
import os
import warnings
import sys
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('ggplot')
sns.set_palette('colorblind')

# Configuration
CSV_DIRECTORY = 'data/stock_prices/'
CLUSTER_FILE = 'data/clusters/cluster_assignments_sweepy_sweep_50.json'
TRAINING_START = '2010-01-01'
TRAINING_END = '2019-01-01'
TESTING_START = '2019-01-01'
TESTING_END = '2024-01-01'
SPY_TICKER = 'SPY'

# Create output directory
output_dir = f"results/results_{os.path.splitext(os.path.basename(CLUSTER_FILE))[0]}"
os.makedirs(output_dir, exist_ok=True)

# Setup logging to file
class Logger:
    def __init__(self, output_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(output_dir, f'analysis_log_{timestamp}.txt')
        self.terminal = sys.stdout
        
    def write(self, message):
        self.terminal.write(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message)
            
    def flush(self):
        self.terminal.flush()

# Redirect stdout to both console and file
logger = Logger(output_dir)
sys.stdout = logger

def load_cluster_assignments(file_path):
    """Load cluster assignments from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        cluster_data = json.load(f)
    
    df = pd.DataFrame(cluster_data['data'], columns=cluster_data['columns'])
    print(f"Loaded {len(df)} cluster assignments")
    return df

def load_spy_data(start_date, end_date, max_retries=3):
    """Load SPY data with retry logic and validation."""
    print(f"\n=== LOADING SPY BENCHMARK DATA ===")
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}: Downloading SPY from {start_date} to {end_date}")
            spy_data = yf.download(SPY_TICKER, start=start_date, end=end_date, 
                                 auto_adjust=True, progress=False)
            
            if spy_data.empty:
                print(f"WARNING: SPY download returned empty DataFrame on attempt {attempt + 1}")
                continue
            
            # Extract Close prices - handle both single and multi-column cases
            if isinstance(spy_data.columns, pd.MultiIndex):
                # Multi-level columns (ticker, price_type)
                spy_prices = spy_data[('Close', SPY_TICKER)] if ('Close', SPY_TICKER) in spy_data.columns else spy_data.iloc[:, 0]
            elif 'Close' in spy_data.columns:
                spy_prices = spy_data['Close']
            else:
                # Single column case
                spy_prices = spy_data.iloc[:, 0]
            
            # Validate data
            spy_prices = spy_prices.dropna()
            
            if len(spy_prices) == 0:
                print(f"ERROR: SPY data is empty after cleaning on attempt {attempt + 1}")
                continue
            
            print(f"✓ SPY data loaded successfully:")
            print(f"  Date range: {spy_prices.index[0].strftime('%Y-%m-%d')} to {spy_prices.index[-1].strftime('%Y-%m-%d')}")
            print(f"  Data points: {len(spy_prices)}")
            print(f"  First value: ${spy_prices.iloc[0]:.2f}")
            print(f"  Last value: ${spy_prices.iloc[-1]:.2f}")
            print(f"  Sample values: {spy_prices.head(3).round(2).tolist()}")
            
            return spy_prices
            
        except Exception as e:
            print(f"ERROR downloading SPY (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                print("CRITICAL: Failed to download SPY after all retries!")
                return None
    
    return None

def load_csv_price_data(csv_directory, tickers, start_date, end_date):
    """Load price data from CSV files only (excludes SPY)."""
    all_data = {}
    missing_files = []
    
    print(f"\n=== LOADING CSV PRICE DATA ===")
    print(f"Loading price data for {len(tickers)} tickers from {start_date} to {end_date}...")
    print(f"Looking for CSV files in: {os.path.abspath(csv_directory)}")
    
    # Check if directory exists
    if not os.path.exists(csv_directory):
        print(f"ERROR: CSV directory does not exist: {csv_directory}")
        return pd.DataFrame(), tickers  # All tickers are missing
    
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files in directory")
    
    for ticker in tickers:
        csv_file = os.path.join(csv_directory, f"{ticker}.csv")
        if not os.path.exists(csv_file):
            missing_files.append(ticker)
            continue
        
        try:
            data = pd.read_csv(csv_file)
            
            # Handle different possible date column names
            date_columns = ['date', 'Date', 'DATE', 'timestamp', 'Timestamp']
            date_col = None
            for col in date_columns:
                if col in data.columns:
                    date_col = col
                    break
            
            if date_col is None:
                print(f"✗ Error loading {ticker}: No date column found. Columns: {list(data.columns)}")
                missing_files.append(ticker)
                continue
            
            data[date_col] = pd.to_datetime(data[date_col])
            data.set_index(date_col, inplace=True)
            
            # Filter date range
            data = data[(data.index >= start_date) & (data.index < end_date)]
            
            if data.empty:
                print(f"✗ No data in date range for {ticker}")
                missing_files.append(ticker)
                continue
            
            # Handle different possible price column names
            price_columns = ['adjusted_close', 'Adjusted_Close', 'adj_close', 'close', 'Close', 'CLOSE']
            price_col = None
            for col in price_columns:
                if col in data.columns:
                    price_col = col
                    break
            
            if price_col is None:
                print(f"✗ Error loading {ticker}: No price column found. Columns: {list(data.columns)}")
                missing_files.append(ticker)
                continue
            
            all_data[ticker] = data[price_col]
            
        except Exception as e:
            print(f"✗ Error loading {ticker}: {e}")
            missing_files.append(ticker)
            continue
    
    if not all_data:
        print("No CSV data loaded successfully")
        return pd.DataFrame(), missing_files
    
    # Combine into DataFrame and resample to monthly
    price_df = pd.concat(all_data.values(), axis=1, keys=all_data.keys())
    price_df = price_df.resample('M').last()
    
    # Only forward fill up to 1 period to handle minor data gaps
    price_df = price_df.fillna(method='ffill', limit=1)
    
    print(f"✓ Successfully loaded CSV data for {len(price_df.columns)} tickers")
    if missing_files:
        print(f"WARNING: {len(missing_files)} tickers had missing CSV files")
    
    return price_df, missing_files

def load_missing_from_yfinance(missing_tickers, start_date, end_date, max_concurrent=10):
    """Load missing tickers from Yahoo Finance as backup."""
    if not missing_tickers:
        return pd.DataFrame()
    
    print(f"\n=== BACKUP LOADING FROM YAHOO FINANCE ===")
    print(f"Attempting to load {len(missing_tickers)} missing tickers from Yahoo Finance...")
    
    yf_data = {}
    successful_loads = 0
    failed_loads = []
    
    # Process in smaller batches to avoid overwhelming Yahoo Finance
    batch_size = min(max_concurrent, len(missing_tickers))
    
    for i in range(0, len(missing_tickers), batch_size):
        batch = missing_tickers[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}: {batch}")
        
        for ticker in batch:
            try:
                # Download individual ticker
                data = yf.download(ticker, start=start_date, end=end_date, 
                                 auto_adjust=True, progress=False)
                
                if data.empty:
                    print(f"  ✗ {ticker}: No data returned")
                    failed_loads.append(ticker)
                    continue
                
                # Extract Close prices - handle both single and multi-column cases
                if isinstance(data.columns, pd.MultiIndex):
                    # Multi-level columns (ticker, price_type)
                    if ('Close', ticker) in data.columns:
                        prices = data[('Close', ticker)]
                    else:
                        prices = data.iloc[:, 0]  # Take first column
                elif 'Close' in data.columns:
                    prices = data['Close']
                else:
                    prices = data.iloc[:, 0]  # Single column case
                
                # Clean and validate
                prices = prices.dropna()
                
                if len(prices) == 0:
                    print(f"  ✗ {ticker}: Empty after cleaning")
                    failed_loads.append(ticker)
                    continue
                
                # Store the data
                yf_data[ticker] = prices
                successful_loads += 1
                print(f"  ✓ {ticker}: {len(prices)} data points ({prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')})")
                
            except Exception as e:
                print(f"  ✗ {ticker}: Error - {e}")
                failed_loads.append(ticker)
                continue
    
    print(f"\nYahoo Finance backup results:")
    print(f"  ✓ Successfully loaded: {successful_loads}")
    print(f"  ✗ Failed to load: {len(failed_loads)}")
    
    if failed_loads:
        print(f"  Failed tickers: {failed_loads[:10]}{'...' if len(failed_loads) > 10 else ''}")
    
    if not yf_data:
        print("No backup data loaded from Yahoo Finance")
        return pd.DataFrame()
    
    # Combine Yahoo Finance data and resample to monthly
    yf_df = pd.concat(yf_data.values(), axis=1, keys=yf_data.keys())
    yf_df = yf_df.resample('M').last()
    
    # Only forward fill up to 1 period to handle minor data gaps
    yf_df = yf_df.fillna(method='ffill', limit=1)
    
    print(f"✓ Yahoo Finance backup DataFrame shape: {yf_df.shape}")
    
    return yf_df

def combine_all_price_data(csv_prices, spy_prices, yf_backup_prices=pd.DataFrame()):
    """Combine CSV price data with SPY data and Yahoo Finance backup data."""
    print(f"\n=== COMBINING ALL PRICE DATA ===")
    
    if spy_prices is None:
        print("ERROR: SPY data is None - proceeding with CSV data only")
        return csv_prices
    
    # Resample SPY to monthly (end of month)
    spy_monthly = spy_prices.resample('M').last()
    
    print(f"CSV data shape: {csv_prices.shape if not csv_prices.empty else 'Empty'}")
    print(f"SPY monthly data points: {len(spy_monthly)}")
    print(f"Yahoo Finance backup shape: {yf_backup_prices.shape if not yf_backup_prices.empty else 'Empty'}")
    print(f"SPY date range: {spy_monthly.index[0].strftime('%Y-%m-%d')} to {spy_monthly.index[-1].strftime('%Y-%m-%d')}")
    
    # Create SPY DataFrame
    spy_df = pd.DataFrame({SPY_TICKER: spy_monthly})
    
    # Start with CSV data
    if not csv_prices.empty:
        combined_data = csv_prices.copy()
    else:
        combined_data = pd.DataFrame()
    
    # Add Yahoo Finance backup data
    if not yf_backup_prices.empty:
        if combined_data.empty:
            combined_data = yf_backup_prices.copy()
        else:
            # Combine CSV and YF backup data
            combined_data = pd.concat([combined_data, yf_backup_prices], axis=1)
        print(f"After adding YF backup: {combined_data.shape}")
    
    # Add SPY data
    if combined_data.empty:
        combined_data = spy_df
    else:
        combined_data = pd.concat([combined_data, spy_df], axis=1)
    
    print(f"Final combined shape: {combined_data.shape}")
    print(f"Final combined columns: {list(combined_data.columns)}")
    print(f"SPY in final combined data: {SPY_TICKER in combined_data.columns}")
    
    # Show sample of combined data
    if not combined_data.empty:
        print(f"Date range: {combined_data.index[0].strftime('%Y-%m-%d')} to {combined_data.index[-1].strftime('%Y-%m-%d')}")
        print("Sample data (first 3 rows):")
        print(combined_data.head(3))
    
    return combined_data

def split_returns_by_period(log_returns, cumulative_returns, training_end):
    """Split returns data into training and testing periods."""
    training_log = log_returns[log_returns.index < training_end]
    testing_log = log_returns[log_returns.index >= training_end]
    
    training_cumulative = cumulative_returns[cumulative_returns.index < training_end]
    testing_cumulative = cumulative_returns[cumulative_returns.index >= training_end]
    
    # Rebase testing cumulative returns to start at 100
    if not testing_cumulative.empty:
        first_values = testing_cumulative.iloc[0]
        # Avoid division by zero - if a portfolio is at 0, keep it at 0
        first_values = first_values.replace(0, 1)  # Treat 0 as 1 to avoid inf
        testing_cumulative = testing_cumulative.div(first_values) * 100
    
    return (training_log, training_cumulative), (testing_log, testing_cumulative)

def calculate_returns(prices):
    """Calculate log returns, imputing missing data until the true end of each stock's life."""
    print(f"\n=== CALCULATING RETURNS ===")
    print(f"Input prices shape: {prices.shape}")
    print(f"Input columns: {list(prices.columns)}")
    
    prices_adjusted = prices.copy()
    
    delisted_count = 0
    for col in prices_adjusted.columns:
        if col == SPY_TICKER:
            # Handle SPY differently - just forward fill missing values
            prices_adjusted[col] = prices_adjusted[col].fillna(method='ffill')
            continue
            
        # Forward fill ALL missing data in the middle (assume data collection issues)
        prices_adjusted[col] = prices_adjusted[col].fillna(method='ffill')
        
        # Find where this stock's data actually ends (last non-null value)
        last_valid_idx = prices_adjusted[col].last_valid_index()
        
        if last_valid_idx is not None and last_valid_idx < prices_adjusted.index[-1]:
            # Stock ended before our analysis period ends
            months_from_end = len(prices_adjusted.index[prices_adjusted.index > last_valid_idx])
            
            # Set price to $0.01 only after the stock truly ends (delisting)
            delisting_mask = prices_adjusted.index > last_valid_idx
            prices_adjusted.loc[delisting_mask, col] = 0.01
            
            delisted_count += 1
    
    if delisted_count > 0:
        print(f"Processed {delisted_count} delisted stocks")
    
    # Fill any remaining NaNs (shouldn't be many after forward fill)
    prices_adjusted = prices_adjusted.fillna(method='ffill')
    prices_adjusted = prices_adjusted.fillna(0.01)  # Fallback for any remaining NaNs
    
    # Calculate log returns
    log_returns = np.log(prices_adjusted / prices_adjusted.shift(1))
    log_returns = log_returns.fillna(0)  # First period has no previous price
    
    # Calculate cumulative returns (portfolio value starting at $100)
    cumulative_returns = np.exp(log_returns.cumsum()) * 100
    
    print(f"Output log returns shape: {log_returns.shape}")
    print(f"Output cumulative returns shape: {cumulative_returns.shape}")
    print(f"SPY in returns: {SPY_TICKER in log_returns.columns}")
    
    if SPY_TICKER in log_returns.columns:
        spy_log = log_returns[SPY_TICKER]
        spy_cum = cumulative_returns[SPY_TICKER]
        print(f"SPY returns validation:")
        print(f"  Log returns count: {len(spy_log.dropna())}")
        print(f"  Cumulative start: {spy_cum.iloc[0]:.2f}")
        print(f"  Cumulative end: {spy_cum.iloc[-1]:.2f}")
    
    return log_returns, cumulative_returns

def create_cluster_portfolios(returns_data, cluster_df, missing_files, max_cluster_size=100):
    """Create equally weighted portfolios for each cluster + SPY benchmark."""
    log_returns, cumulative_returns = returns_data
    available_tickers = set(log_returns.columns)
    
    print(f"\n=== CREATING CLUSTER PORTFOLIOS ===")
    print(f"Available tickers for portfolios: {len(available_tickers)}")
    print(f"SPY_TICKER = '{SPY_TICKER}'")
    print(f"SPY in available tickers: {SPY_TICKER in available_tickers}")
    
    portfolios = {'log': {}, 'cumulative': {}, 'constituents': {}}
    
    # CRITICAL: Add SPY benchmark FIRST
    if SPY_TICKER in available_tickers:
        portfolios['log']['SP500'] = log_returns[SPY_TICKER].copy()
        portfolios['cumulative']['SP500'] = cumulative_returns[SPY_TICKER].copy()
        portfolios['constituents']['SP500'] = [SPY_TICKER]
        print(f"✓ Successfully added S&P 500 benchmark ({SPY_TICKER})")
        
        # Validate SPY portfolio data
        spy_log = portfolios['log']['SP500']
        spy_cum = portfolios['cumulative']['SP500']
        print(f"  SPY portfolio validation:")
        print(f"    Log returns - count: {len(spy_log.dropna())}, mean: {spy_log.mean():.6f}")
        print(f"    Cumulative - start: {spy_cum.iloc[0]:.2f}, end: {spy_cum.iloc[-1]:.2f}")
        
    else:
        print(f"✗ CRITICAL ERROR: {SPY_TICKER} not found in available tickers!")
        print(f"  Available tickers: {sorted(list(available_tickers))}")
        print("  SPY benchmark will not be available!")
    
    # Analyze cluster sizes
    cluster_sizes = cluster_df['cluster'].value_counts()
    print(f"\nCluster sizes: {dict(cluster_sizes.sort_index())}")
    
    # Create cluster portfolios (exclude clusters 0, -1, and very large clusters)
    portfolio_missing_files = {}
    
    for cluster in sorted(cluster_df['cluster'].unique()):
        if cluster in [-1]:
            continue
        
        cluster_size = cluster_sizes[cluster]
        if cluster_size > max_cluster_size:
            print(f"Skipping Cluster_{cluster} (size: {cluster_size}) - too large, likely generic")
            continue
        
        cluster_tickers = cluster_df[cluster_df['cluster'] == cluster]['ticker'].tolist()
        available_cluster_tickers = [t for t in cluster_tickers if t in available_tickers and t != SPY_TICKER]
        
        # Check for missing files in this cluster
        missing_in_cluster = [t for t in cluster_tickers if t in missing_files]
        
        if len(available_cluster_tickers) == 0:
            if missing_in_cluster:
                print(f"ERROR: Cluster_{cluster} has NO available tickers!")
                print(f"  Missing CSV files: {missing_in_cluster}")
            continue
        
        portfolio_name = f"Cluster_{cluster}"
        
        # Log missing files for portfolios we're actually creating
        if missing_in_cluster:
            portfolio_missing_files[portfolio_name] = missing_in_cluster
            print(f"WARNING: {portfolio_name} missing {len(missing_in_cluster)} CSV files: {missing_in_cluster}")
        
        # Calculate equally weighted log returns (arithmetic mean in log space)
        portfolio_log_returns = log_returns[available_cluster_tickers].mean(axis=1)
        
        # Calculate cumulative returns from portfolio log returns
        portfolio_cumulative = np.exp(portfolio_log_returns.cumsum()) * 100
        
        portfolios['log'][portfolio_name] = portfolio_log_returns
        portfolios['cumulative'][portfolio_name] = portfolio_cumulative
        portfolios['constituents'][portfolio_name] = available_cluster_tickers
        
        print(f"✓ Created {portfolio_name} with {len(available_cluster_tickers)}/{len(cluster_tickers)} tickers")
    
    # Final validation
    final_portfolios = list(portfolios['log'].keys())
    print(f"\n=== FINAL PORTFOLIO VALIDATION ===")
    print(f"Total portfolios created: {len(final_portfolios)}")
    print(f"Portfolio names: {final_portfolios}")
    print(f"SP500 in final portfolios: {'SP500' in final_portfolios}")
    
    # Summary of missing files impact
    if portfolio_missing_files:
        print(f"\n=== MISSING FILES SUMMARY ===")
        total_missing = sum(len(files) for files in portfolio_missing_files.values())
        print(f"Total missing files affecting portfolios: {total_missing}")
        for portfolio, missing in portfolio_missing_files.items():
            print(f"  {portfolio}: {len(missing)} missing files")
    
    # Convert to DataFrames
    portfolio_log_df = pd.DataFrame(portfolios['log'])
    portfolio_cumulative_df = pd.DataFrame(portfolios['cumulative'])
    
    print(f"\nFinal DataFrame shapes:")
    print(f"  Log returns: {portfolio_log_df.shape}")
    print(f"  Cumulative returns: {portfolio_cumulative_df.shape}")
    print(f"  Columns: {list(portfolio_log_df.columns)}")
    
    return (portfolio_log_df, portfolio_cumulative_df, portfolios['constituents'])

def calculate_performance_metrics(log_returns, risk_free_rate=0.02/12):
    """Calculate performance metrics using log returns only."""
    metrics = {}
    
    for portfolio in log_returns.columns:
        log_ret = log_returns[portfolio].dropna()
        
        if len(log_ret) < 12:  # Need at least a year of data
            continue
        
        # Total return (convert from log space)
        total_log_return = log_ret.sum()
        total_return = np.exp(total_log_return) - 1
        
        # Annualized return
        years = len(log_ret) / 12
        annualized_return = np.exp(log_ret.mean() * 12) - 1
        
        # Volatility (annualized)
        volatility = log_ret.std() * np.sqrt(12)
        
        # Sharpe ratio (using log returns throughout)
        log_rf_rate = np.log(1 + risk_free_rate)
        excess_return = log_ret.mean() - log_rf_rate
        sharpe_ratio = (excess_return / log_ret.std()) * np.sqrt(12) if log_ret.std() != 0 else 0
        
        # Maximum drawdown (convert to cumulative prices first)
        cumulative = np.exp(log_ret.cumsum())
        max_drawdown = (cumulative / cumulative.cummax() - 1).min()
        
        metrics[portfolio] = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Annualized Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Maximum Drawdown': max_drawdown
        }
    
    return pd.DataFrame(metrics).T

def plot_cumulative_returns(cumulative_df, title, output_path):
    """Plot cumulative returns with appropriate scaling for comparison."""
    print(f"\n=== PLOTTING: {title} ===")
    print(f"Columns to plot: {list(cumulative_df.columns)}")
    print(f"SP500 in plot data: {'SP500' in cumulative_df.columns}")
    
    if cumulative_df.empty:
        print("ERROR: No data to plot!")
        return
    
    plt.figure(figsize=(14, 8))
    
    # Check value ranges to handle scale differences
    max_values = cumulative_df.max()
    min_values = cumulative_df.min()
    
    print(f"Value ranges in plot:")
    for col in cumulative_df.columns:
        print(f"  {col}: {min_values[col]:.1f} to {max_values[col]:.1f}")
    
    # Detect if we have extreme scale differences
    overall_max = max_values.max()
    overall_min = max_values.min()
    scale_ratio = overall_max / overall_min if overall_min > 0 else float('inf')
    
    if scale_ratio > 10:  # Large scale differences
        print(f"WARNING: Large scale differences detected (ratio: {scale_ratio:.1f})")
    
    # Plot S&P 500 first (if available) with distinctive style
    spy_plotted = False
    if 'SP500' in cumulative_df.columns:
        spy_data = cumulative_df['SP500'].dropna()
        if len(spy_data) > 0:
            plt.plot(spy_data.index, spy_data.values, 'k--', 
                    linewidth=3, label='S&P 500', alpha=0.9, zorder=10)
            print("✓ Plotted S&P 500 line successfully")
            spy_plotted = True
        else:
            print("✗ S&P 500 data is empty - cannot plot")
    else:
        print("✗ S&P 500 not found in data - will not appear on plot")
    
    # Plot cluster portfolios
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(cumulative_df.columns), 10)))
    color_idx = 0
    cluster_count = 0
    
    for portfolio in cumulative_df.columns:
        if portfolio == 'SP500':
            continue  # Already plotted above
        
        portfolio_data = cumulative_df[portfolio].dropna()
        if len(portfolio_data) == 0:
            print(f"✗ Skipping {portfolio} - no data")
            continue
            
        label = portfolio.replace('Cluster_', 'Portfel ')
        plt.plot(portfolio_data.index, portfolio_data.values, 
                linewidth=2, label=label, alpha=0.8, color=colors[color_idx])
        color_idx += 1
        cluster_count += 1
        print(f"✓ Plotted {portfolio}")
    
    print(f"Total lines plotted: {cluster_count + (1 if spy_plotted else 0)}")
    
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Data', fontsize=14)
    plt.ylabel('Wartość Portfela (początkowa wartość = 100)', fontsize=14)
    
    # Position legend outside plot area
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Add reference lines
    plt.axhline(y=100, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # Use log scale if extreme differences
    if scale_ratio > 50:
        plt.yscale('log')
        plt.ylabel('Wartość Portfela (skala logarytmiczna, początkowa = 100)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plot to {output_path}")

def plot_risk_return(metrics_df, title, output_path):
    """Create risk-return scatter plot with dynamic axis limits."""
    plt.figure(figsize=(10, 8))
    
    # Extract x and y values
    x_values = metrics_df['Annualized Volatility'].values
    y_values = metrics_df['Annualized Return'].values
    
    for portfolio in metrics_df.index:
        x = metrics_df.loc[portfolio, 'Annualized Volatility']
        y = metrics_df.loc[portfolio, 'Annualized Return']
        
        # Convert display name
        display_name = portfolio.replace('Cluster_', 'Portfel ') if portfolio != 'SP500' else 'S&P 500'
        
        if portfolio == 'SP500':
            plt.scatter(x, y, s=150, color='black', marker='*', label='S&P 500')
        else:
            plt.scatter(x, y, s=100, alpha=0.7)
        
        plt.annotate(display_name, (x, y), xytext=(5, 5), textcoords='offset points')
    
    # Calculate dynamic axis limits with padding
    x_min, x_max = x_values.min(), x_values.max()
    y_min, y_max = y_values.min(), y_values.max()
    
    # Add padding (10% of range on each side)
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    x_padding = max(x_range * 0.1, 0.01)  # At least 1% padding
    y_padding = max(y_range * 0.1, 0.005)  # At least 0.5% padding
    
    # Set axis limits with padding
    plt.xlim(x_min - x_padding, x_max + x_padding)
    plt.ylim(y_min - y_padding, y_max + y_padding)
    
    # Add reference lines only if they're within the visible range
    current_xlim = plt.xlim()
    current_ylim = plt.ylim()
    
    if current_ylim[0] <= 0 <= current_ylim[1]:
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    if current_xlim[0] <= 0 <= current_xlim[1]:
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Roczna Zmienność', fontsize=14)
    plt.ylabel('Roczny Zwrot', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_correlation_heatmap(returns_df, title, output_path):
    """Plot correlation heatmap."""
    plt.figure(figsize=(10, 8))
    
    corr_matrix = returns_df.corr()
    
    # Rename for Polish labels
    corr_matrix.columns = [col.replace('Cluster_', 'Portfel ') if col != 'SP500' else 'S&P 500' 
                          for col in corr_matrix.columns]
    corr_matrix.index = corr_matrix.columns
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                fmt='.2f', linewidths=0.5)
    
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    print("Starting portfolio analysis with SPY benchmark...")
    print(f"SPY_TICKER = '{SPY_TICKER}'")
    
    # Load cluster assignments
    cluster_df = load_cluster_assignments(CLUSTER_FILE)
    
    # Get CSV tickers (excluding clusters 0 and -1) - SPY handled separately
    relevant_clusters = cluster_df[~cluster_df['cluster'].isin([-1])]
    csv_tickers = list(set(relevant_clusters['ticker'].tolist()))
    
    print(f"CSV tickers to load: {len(csv_tickers)}")
    print(f"SPY will be loaded separately as benchmark")
    
    # Load SPY data separately (always as benchmark) - CRITICAL REQUIREMENT
    spy_data = load_spy_data(TRAINING_START, TESTING_END)
    
    # Exit if SPY data cannot be loaded
    if spy_data is None:
        print("\n" + "="*70)
        print("CRITICAL ERROR: SPY benchmark data could not be downloaded!")
        print("Analysis cannot proceed without SPY benchmark.")
        print("Please check your internet connection and try again.")
        print("="*70)
        sys.exit(1)  # Exit with error code
    
    # Load CSV price data (no SPY here)
    csv_prices, missing_files = load_csv_price_data(CSV_DIRECTORY, csv_tickers, 
                                                   TRAINING_START, TESTING_END)
    
    # Try to load missing tickers from Yahoo Finance as backup
    yf_backup_prices = pd.DataFrame()  # Initialize empty DataFrame
    if missing_files:
        print(f"\nAttempting to load {len(missing_files)} missing tickers from Yahoo Finance as backup...")
        yf_backup_prices = load_missing_from_yfinance(missing_files, TRAINING_START, TESTING_END)
        
        # Update missing files list to only include those that failed both CSV and YF
        if not yf_backup_prices.empty:
            successfully_loaded_from_yf = list(yf_backup_prices.columns)
            original_missing_count = len(missing_files)
            missing_files = [ticker for ticker in missing_files if ticker not in successfully_loaded_from_yf]
            print(f"Yahoo Finance backup recovered {original_missing_count - len(missing_files)} tickers")
            print(f"After Yahoo Finance backup: {len(missing_files)} tickers still missing")
        else:
            print("No tickers successfully loaded from Yahoo Finance backup")
    else:
        print("No missing CSV files - Yahoo Finance backup not needed")
    
    # Combine CSV, Yahoo Finance backup, and SPY data
    all_prices = combine_all_price_data(csv_prices, spy_data, yf_backup_prices)
    
    if all_prices.empty:
        print("CRITICAL ERROR: No price data loaded!")
        return
    
    print(f"\n=== FINAL PRICE DATA CHECK ===")
    print(f"All prices shape: {all_prices.shape}")
    print(f"All prices columns: {list(all_prices.columns)}")
    print(f"SPY in final data: {SPY_TICKER in all_prices.columns}")
    
    # Calculate returns for full period
    all_log_returns, all_cumulative_returns = calculate_returns(all_prices)
    
    if all_log_returns.empty:
        print("CRITICAL ERROR: No returns calculated!")
        return
    
    # Create portfolios (includes SPY benchmark)
    portfolio_log_returns, portfolio_cumulative_returns, portfolio_constituents = create_cluster_portfolios(
        (all_log_returns, all_cumulative_returns), cluster_df, missing_files
    )
    
    if portfolio_log_returns.empty:
        print("CRITICAL ERROR: No portfolios created!")
        return
    
    print(f"\n=== PORTFOLIO CREATION COMPLETE ===")
    print(f"Final portfolio columns: {list(portfolio_log_returns.columns)}")
    print(f"SP500 in portfolios: {'SP500' in portfolio_log_returns.columns}")
    
    # Split portfolio returns into training and testing periods
    training_data, testing_data = split_returns_by_period(
        portfolio_log_returns, portfolio_cumulative_returns, TESTING_START
    )
    
    training_log, training_cumulative = training_data
    testing_log, testing_cumulative = testing_data
    
    print(f"\n=== PERIOD SPLIT COMPLETE ===")
    print(f"Training columns: {list(training_cumulative.columns)}")
    print(f"Testing columns: {list(testing_cumulative.columns)}")
    print(f"SP500 in training: {'SP500' in training_cumulative.columns}")
    print(f"SP500 in testing: {'SP500' in testing_cumulative.columns}")
    
    # Calculate performance metrics for each period
    training_metrics = calculate_performance_metrics(training_log)
    testing_metrics = calculate_performance_metrics(testing_log)
    
    # Save metrics
    training_metrics.to_csv(os.path.join(output_dir, 'training_metrics.csv'))
    testing_metrics.to_csv(os.path.join(output_dir, 'testing_metrics.csv'))
    
    # Generate plots
    print("\n=== GENERATING VISUALIZATIONS ===")
    
    plot_cumulative_returns(
        training_cumulative, 
        'Wartość portfeli w okresie treningowym (2010-2019)',
        os.path.join(output_dir, 'training_performance.png')
    )
    
    plot_cumulative_returns(
        testing_cumulative, 
        'Wartość portfeli w okresie testowym (2019-2024)',
        os.path.join(output_dir, 'testing_performance.png')
    )
    
    plot_risk_return(
        training_metrics,
        'Zwrot vs. Ryzyko (Okres Treningowy: 2010-2019)',
        os.path.join(output_dir, 'training_risk_return.png')
    )
    
    plot_risk_return(
        testing_metrics,
        'Zwrot vs Ryzyko (Okres Testowy: 2019-2024)',
        os.path.join(output_dir, 'testing_risk_return.png')
    )
    
    plot_correlation_heatmap(
        training_log,
        'Korelacja zwrotów portfeli w okresie treningowym (2010-2019)',
        os.path.join(output_dir, 'training_correlation.png')
    )
    
    plot_correlation_heatmap(
        testing_log,
        'Korelacja zwrotów portfeli w okresie testowym (2019-2024)',
        os.path.join(output_dir, 'testing_correlation.png')
    )
    
    # Create performance summary
    summary = pd.concat([
        training_metrics[['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio']].add_prefix('Training_'),
        testing_metrics[['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio']].add_prefix('Testing_')
    ], axis=1)
    
    summary.to_csv(os.path.join(output_dir, 'performance_summary.csv'))
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Results saved to '{output_dir}' directory.")
    
    # Final validation
    if 'SP500' in portfolio_log_returns.columns:
        print("✓ SUCCESS: SPY benchmark successfully included in all outputs!")
        print("✓ SPY should now appear in all your graphs as 'S&P 500'")
    else:
        print("✗ FAILURE: SPY benchmark missing - check internet connection and yfinance installation")
    
    # Show what was created
    print(f"\nFiles created:")
    for filename in ['training_performance.png', 'testing_performance.png', 
                     'training_risk_return.png', 'testing_risk_return.png',
                     'training_correlation.png', 'testing_correlation.png',
                     'training_metrics.csv', 'testing_metrics.csv', 
                     'performance_summary.csv']:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} (not created)")
    
    print(f"\nLog file saved to: {logger.log_file}")

if __name__ == "__main__":
    main()