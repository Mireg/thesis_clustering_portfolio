import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Unit standardization and robust handling of outliers
def standardize_financial_values(df):
    """
    Standardize financial values by detecting and fixing unit inconsistencies
    and extreme outliers.
    """
    print("Standardizing financial values and fixing inconsistencies...")
    
    standardized_df = df.copy()
    financial_cols = [col for col in df.columns if col not in ['ticker', 'date']]
    
    # Process each company separately
    for ticker, company_data in df.groupby('ticker'):
        # Sort by date to ensure proper time sequence
        company_data = company_data.sort_values('date')
        company_idx = company_data.index
        
        for col in financial_cols:
            series = company_data[col].copy()
            # Skip columns with too many missing values
            if series.isna().sum() > 0.8 * len(series):
                continue
                
            # Get non-null values
            values = series.dropna().values
            if len(values) < 3:  # Need at least 3 points for meaningful analysis
                continue
            
            # Calculate orders of magnitude
            non_zero_values = values[values != 0]
            if len(non_zero_values) < 3:
                continue
                
            magnitudes = np.log10(np.abs(non_zero_values))
            median_magnitude = np.median(magnitudes)
            
            # Detect values with vastly different magnitudes (potential unit issues)
            magnitude_diffs = np.abs(magnitudes - median_magnitude)
            # Values that differ by more than 5 orders of magnitude are suspect
            suspicious_mask = magnitude_diffs > 5
            
            if suspicious_mask.any():
                # For each suspicious value, try to fix it
                for i, is_suspicious in enumerate(suspicious_mask):
                    if is_suspicious:
                        idx = np.where(values == non_zero_values[i])[0][0]
                        orig_value = values[idx]
                        # If magnitude is too high, divide by 1M
                        if magnitudes[i] > median_magnitude + 5:
                            values[idx] = orig_value / 1000000
                        # If magnitude is too low, multiply by 1M
                        elif magnitudes[i] < median_magnitude - 5:
                            values[idx] = orig_value * 1000000
                
                # Update the standardized dataframe
                non_null_indices = company_data.index[~company_data[col].isna()]
                for i, idx in enumerate(non_null_indices):
                    if i < len(values):
                        standardized_df.loc[idx, col] = values[i]
    
    return standardized_df

def detect_and_fix_outliers(df, z_threshold=3.5):
    """
    Detect and fix extreme outliers using z-scores and rolling medians.
    """
    print("Detecting and fixing extreme outliers...")
    
    cleaned_df = df.copy()
    financial_cols = [col for col in df.columns if col not in ['ticker', 'date']]
    
    for ticker, company_data in df.groupby('ticker'):
        company_data = company_data.sort_values('date')
        
        for col in financial_cols:
            series = company_data[col].copy()
            if series.isna().sum() > 0.7 * len(series):
                continue
                
            # Calculate z-scores (ignoring NaNs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
            
            # Identify extreme outliers
            mask = ~series.isna()
            outlier_mask = mask & (z_scores > z_threshold)
            
            if outlier_mask.any():
                # Calculate rolling median with window size 3
                rolling_med = series.rolling(window=3, min_periods=1, center=True).median()
                
                # Replace outliers with rolling median
                cleaned_df.loc[company_data.index[outlier_mask], col] = rolling_med[outlier_mask]
    
    return cleaned_df

def log_transform_safe(x, offset=0.01):
    """
    Apply log transformation safely handling zeros and negative values.
    """
    if pd.isna(x):
        return np.nan
    elif x > 0:
        return np.log(x + offset)
    elif x < 0:
        return -np.log(abs(x) + offset)
    else:
        return 0

# Improved safe_divide function
def safe_divide(a, b, fallback=np.nan, bounds=None):
    """
    Safely divide with protection against infinite values
    but preserve NaN behavior for missing values.
    
    bounds: optional tuple (min_val, max_val) to clip results
    """
    result = a / b
    # Replace infinities, not NaNs
    result = result.replace([np.inf, -np.inf], fallback)
    
    if bounds is not None:
        min_val, max_val = bounds
        result = result.clip(min_val, max_val)
        
    return result

# Data is NMAR, shouldn't ignore missing values
def analyze_missing_patterns(df):
    """
    Analyze and create features from missing data patterns
    """
    print("Analyzing missing patterns...")
    
    # Define key metric groups
    key_metrics = {
        'structural': ['RnD', 'Goodwill', 'LongTermDebt'],
        'core': ['Revenue', 'NetIncome', 'Assets', 'OperatingIncome'],
        'operational': ['CapEx', 'CurrentAssets', 'CurrentLiabilities']
    }
    
    companies = df.groupby('ticker')
    patterns = {}
    
    for ticker, company_data in companies:
        patterns[ticker] = {}
        
        # 1. Structural patterns (never/always missing)
        for metric in key_metrics['structural']:
            patterns[ticker][f'has_{metric.lower()}'] = not company_data[metric].isna().all()
        
        # 2. Core metric completeness
        core_completeness = 1 - company_data[key_metrics['core']].isna().mean().mean()
        patterns[ticker]['core_completeness'] = core_completeness
        
        # 3. Reporting consistency
        patterns[ticker]['reporting_consistency'] = 1 - (company_data[key_metrics['core']].isna().sum().sum() / 
                                                       (len(company_data) * len(key_metrics['core'])))
    
    return pd.DataFrame.from_dict(patterns, orient='index')

def calculate_financial_ratios(df):
    """Calculate financial ratios with protection against infinite values and unrealistic values"""
    print("Calculating financial ratios...")
    
    ratios = pd.DataFrame(index=df.index)
    
    # Profitability ratios with reasonable bounds
    ratios['ROA'] = safe_divide(df['NetIncome'], df['Assets'], bounds=(-1, 1))
    ratios['ROE'] = safe_divide(df['NetIncome'], df['StockholdersEquity'], bounds=(-2, 2))
    ratios['OperatingMargin'] = safe_divide(df['OperatingIncome'], df['Revenue'], bounds=(-1, 1))
    
    # Efficiency ratios
    ratios['AssetTurnover'] = safe_divide(df['Revenue'], df['Assets'], bounds=(0, 10))
    ratios['CurrentRatio'] = safe_divide(df['CurrentAssets'], df['CurrentLiabilities'], fallback=1, bounds=(0, 10))
    
    # Leverage ratios
    ratios['Leverage'] = safe_divide(df['Liabilities'], df['Assets'], bounds=(0, 5))
    ratios['DebtToEquity'] = safe_divide(df['LongTermDebt'], df['StockholdersEquity'], bounds=(-10, 10))
    
    return ratios

def create_log_transformed_features(df):
    """
    Create log-transformed versions of key financial metrics
    to reduce the impact of scale differences.
    """
    print("Creating log-transformed features...")
    
    log_features = pd.DataFrame(index=df.index)
    
    # Apply log transformation to key financial metrics
    for col in ['Revenue', 'Assets', 'NetIncome', 'OperatingIncome', 'Liabilities', 
               'StockholdersEquity', 'CurrentAssets', 'CurrentLiabilities', 
               'LongTermDebt', 'CapEx']:
        if col in df.columns:
            log_features[f'log_{col}'] = df[col].apply(log_transform_safe)
    
    return log_features

def create_optimized_features(df, ratios, log_features):
    """
    Create optimized feature set:
    - Log transforms for absolute metrics
    - Regular values for ratios
    - Growth rates and volatility measures
    """
    print("Creating optimized features...")
    
    companies = df.groupby('ticker')
    features = {}
    
    for ticker, company_data in companies:
        features[ticker] = {}
        
        # 1. Average ratios from original data
        for col in ratios.columns:
            company_ratios = ratios.loc[company_data.index, col]
            # Replace inf with appropriate fallback before calculating mean
            clean_ratios = company_ratios.replace([np.inf, -np.inf], np.nan)
            features[ticker][f'avg_{col}'] = clean_ratios.mean()
            
            # Need at least 2 values for std calculation
            if len(clean_ratios.dropna()) > 1:
                features[ticker][f'{col}_volatility'] = clean_ratios.std()
            else:
                features[ticker][f'{col}_volatility'] = np.nan
        
        # 2. Log-transformed features (for absolute values)
        for col in log_features.columns:
            log_vals = log_features.loc[company_data.index, col].dropna()
            if len(log_vals) > 0:
                features[ticker][f'avg_{col}'] = log_vals.mean()
                if len(log_vals) > 1:
                    features[ticker][f'{col}_volatility'] = log_vals.std()
        
        # 3. Growth rates with robust calculation
        for metric in ['Revenue', 'Assets', 'NetIncome']:
            if not company_data[metric].isna().all():
                clean_data = company_data[metric].dropna()
                if len(clean_data) > 1:
                    start_val = clean_data.iloc[0]
                    end_val = clean_data.iloc[-1]
                    # Check for valid values for growth calculation
                    if start_val > 0 and end_val > 0:
                        # Use log difference for more robust growth calculation
                        time_periods = len(clean_data) - 1
                        growth_rate = (np.log(end_val) - np.log(start_val)) / time_periods
                        growth_rate = np.exp(growth_rate) - 1
                        
                        # Apply reasonable bounds
                        if not np.isinf(growth_rate) and abs(growth_rate) <= 1:
                            features[ticker][f'{metric}_growth'] = growth_rate
                        else:
                            features[ticker][f'{metric}_growth'] = np.nan
                    else:
                        features[ticker][f'{metric}_growth'] = np.nan
                else:
                    features[ticker][f'{metric}_growth'] = np.nan
            else:
                features[ticker][f'{metric}_growth'] = np.nan
        
        # 4. Capital intensity
        if not company_data['CapEx'].isna().all() and not company_data['Revenue'].isna().all():
            capex_ratios = safe_divide(company_data['CapEx'], company_data['Revenue'], bounds=(0, 1))
            features[ticker]['capex_intensity'] = capex_ratios.mean()
        else:
            features[ticker]['capex_intensity'] = np.nan
            
    return pd.DataFrame.from_dict(features, orient='index')

def impute_final_features(feature_df, pattern_df):
    """
    Impute missing values in final feature set
    """
    print("Performing final imputation...")
    
    df = feature_df.copy()
    
    # 0. Replace any infinities
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 1. Size-based grouping for more accurate imputation
    log_assets_col = next((col for col in df.columns if col == 'avg_log_Assets'), None)
    if log_assets_col:
        df[log_assets_col] = df[log_assets_col].fillna(df[log_assets_col].median())
        df['size_category'] = pd.qcut(df[log_assets_col], q=5, labels=['vs', 's', 'm', 'l', 'vl'])
    else:
        df['size_category'] = 'm'  # Default if log_assets not available
    
    # 2. Feature types for different imputation strategies
    log_features = [col for col in df.columns if 'log_' in col and not 'volatility' in col]
    ratio_features = [col for col in df.columns if col.startswith('avg_') and not 'log_' in col]
    volatility_features = [col for col in df.columns if 'volatility' in col]
    growth_features = [col for col in df.columns if 'growth' in col]
    operational_features = ['capex_intensity']
    
    # 3. Impute different feature groups
    # Log-transformed features - median within size category
    for col in log_features:
        if df[col].isna().any():
            df[col] = df.groupby('size_category')[col].transform(lambda x: x.fillna(x.median()))
    
    # Ratio features - median within size category
    for col in ratio_features:
        if df[col].isna().any():
            df[col] = df.groupby('size_category')[col].transform(lambda x: x.fillna(x.median()))
    
    # Volatility features - median imputation
    for col in volatility_features:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Growth features - 0 imputation with flag
    for col in growth_features:
        if df[col].isna().any():
            df[f'{col}_imputed'] = df[col].isna().astype(int)
            df[col] = df[col].fillna(0)
    
    # Operating metrics - median within size category with flag
    for col in operational_features:
        if col in df.columns and df[col].isna().any():
            df[f'{col}_imputed'] = df[col].isna().astype(int)
            df[col] = df.groupby('size_category')[col].transform(lambda x: x.fillna(x.median()))
    
    # 4. Remove temporary column
    df = df.drop('size_category', axis=1)
    
    # 5. Combine with pattern features
    df = pd.concat([df, pattern_df], axis=1)
    
    # 6. Final check for infinities
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 7. If any NaNs remain, use median imputation
    if df.isna().sum().sum() > 0:
        print("Warning: Some missing values remain - applying final median imputation")
        df = df.fillna(df.median())
    
    return df

def verify_distributions(df):
    """
    Print summary statistics for features to verify distributions.
    """
    print("\nVerifying feature distributions...")
    
    print("\nMeans:")
    print(df.mean().sort_values())
    
    print("\nStandard deviations:")
    print(df.std().sort_values())
    
    print("\nMin/Max values:")
    min_max = pd.DataFrame({
        'min': df.min(),
        'max': df.max()
    })
    print(min_max.sort_values('max', ascending=False))
    
    return df

def create_final_dataset(df):
    """
    Main function to create final feature set with robust methods and reduced redundancy
    """
    # 0. Standardize values and fix inconsistencies
    df = standardize_financial_values(df)
    df = detect_and_fix_outliers(df)
    
    # 1. Analyze missing patterns
    pattern_features = analyze_missing_patterns(df)
    
    # 2. Calculate raw ratios
    raw_ratios = calculate_financial_ratios(df)
    
    # 3. Create log-transformed features
    log_features = create_log_transformed_features(df)
    
    # 4. Create optimized features (combines the best of both)
    optimized_features = create_optimized_features(df, raw_ratios, log_features)
    
    # 5. Final imputation and combination
    final_features = impute_final_features(optimized_features, pattern_features)
    
    # 6. Verify distributions
    final_features = verify_distributions(final_features)
    
    # Print feature summary
    print("\nFinal dataset shape:", final_features.shape)
    
    # Count features by type
    log_feature_count = len([col for col in final_features.columns if 'log_' in col and not 'volatility' in col])
    ratio_feature_count = len([col for col in final_features.columns if col.startswith('avg_') and not 'log_' in col])
    volatility_count = len([col for col in final_features.columns if 'volatility' in col])
    growth_count = len([col for col in final_features.columns if 'growth' in col and not 'imputed' in col])
    pattern_count = len([col for col in final_features.columns if col in pattern_features.columns])
    imputed_count = len([col for col in final_features.columns if 'imputed' in col])
    
    print("\nFeature counts by type:")
    print(f"Log-transformed metrics: {log_feature_count}")
    print(f"Financial ratios: {ratio_feature_count}")
    print(f"Volatility metrics: {volatility_count}")
    print(f"Growth metrics: {growth_count}")
    print(f"Missing pattern metrics: {pattern_count}")
    print(f"Imputation flags: {imputed_count}")
    
    return final_features

if __name__ == '__main__':
    # Load the data
    print("Loading data...")
    df = pd.read_csv('./data/intermediate/pivoted_financial_data_2010.csv')

    # Trim the data to required period
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].dt.year >= 2010]
    df = df[df['date'].dt.year < 2019]

    # Run the pipeline
    final_df = create_final_dataset(df)
    
    print("\nMissing values after processing:")
    print(final_df.isna().sum()[final_df.isna().sum() > 0])
    
    # Check for infinities in final dataset
    inf_check = (final_df == np.inf) | (final_df == -np.inf)
    if inf_check.values.any():
        print("\nWarning: Infinities found in final dataset!")
        print(final_df.columns[inf_check.any()].tolist())
    else:
        print("\nNo infinities found in final dataset.")

    # Adjust datatypes and save the file 
    for col in final_df.columns:
        if col.startswith('has_') or col.endswith('_imputed'):
            final_df[col] = final_df[col].astype(int)

    file_path = './data/processed/financial_features.csv'
    final_df.to_csv(file_path)
    
    print(f'\nSaved optimized features to {file_path}')