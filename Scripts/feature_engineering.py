import pandas as pd
import numpy as np

# Data is NMAR, shouldn't ignore missing values
def analyze_missing_patterns(df):
    """
    Analyze and create features from missing data patterns
    """
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
        
        # 3. Reporting patterns
        for metric in key_metrics['core'] + key_metrics['operational']:
            metric_data = company_data[metric].reset_index(drop=True)
            not_null = ~metric_data.isna()
            if not_null.any():
                first_reported = not_null.idxmax()
                # Calculate when reporting starts as a percentage of total time
                patterns[ticker][f'{metric}_start'] = first_reported / len(metric_data)
            else:
                patterns[ticker][f'{metric}_start'] = -1
        
        # 4. Overall reporting quality
        patterns[ticker]['reporting_consistency'] = 1 - (company_data[key_metrics['core']].isna().sum().sum() / 
                                                       (len(company_data) * len(key_metrics['core'])))
    
    return pd.DataFrame.from_dict(patterns, orient='index')

# Don't want to impute at this stage yet, want to keep information from missing values and prevent data leakage. 
def calculate_raw_ratios(df):
    """
    Calculate financial ratios (without imputation)
    """
    ratios = pd.DataFrame(index=df.index)
    
    # Profitability ratios
    ratios['ROA'] = df['NetIncome'] / df['Assets']
    ratios['ROE'] = df['NetIncome'] / df['StockholdersEquity']
    ratios['OperatingMargin'] = df['OperatingIncome'] / df['Revenue']
    
    # Efficiency ratios
    ratios['AssetTurnover'] = df['Revenue'] / df['Assets']
    ratios['CurrentRatio'] = df['CurrentAssets'] / df['CurrentLiabilities']
    
    # Leverage ratios
    ratios['Leverage'] = df['Liabilities'] / df['Assets']
    ratios['DebtToEquity'] = df['LongTermDebt'] / df['StockholdersEquity']
    
    return ratios

# We need only one value per feature per company, dont want to deal with time series and rolling windows. 
def create_base_features(df, ratios):
    """
    Create consolidated features while preserving missing values.
    """
    companies = df.groupby('ticker')
    features = {}
    
    for ticker, company_data in companies:
        features[ticker] = {}
        
        # 1. Average ratios (no imputation)
        for col in ratios.columns:
            company_ratios = ratios.loc[company_data.index, col]
            features[ticker][f'avg_{col}'] = company_ratios.mean()
            features[ticker][f'{col}_volatility'] = company_ratios.std()
        
        # 2. Growth rates
        for metric in ['Revenue', 'Assets', 'NetIncome']:
            if not company_data[metric].isna().all():
                clean_data = company_data[metric].dropna()
                if len(clean_data) > 1:
                    start_val = clean_data.iloc[0]
                    end_val = clean_data.iloc[-1]
                    if start_val > 0 and end_val > 0:
                        features[ticker][f'{metric}_growth'] = (end_val/start_val)**(1/len(clean_data)) - 1
                    else:
                        features[ticker][f'{metric}_growth'] = np.nan
                else:
                    features[ticker][f'{metric}_growth'] = np.nan
            else:
                features[ticker][f'{metric}_growth'] = np.nan
        
        # 3. Size and scale
        features[ticker]['log_assets'] = np.log(company_data['Assets'].mean()) if not company_data['Assets'].isna().all() else np.nan
        
        # 4. Operating characteristics
        if not company_data['CapEx'].isna().all() and not company_data['Revenue'].isna().all():
            features[ticker]['capex_intensity'] = (company_data['CapEx'] / company_data['Revenue']).mean()
        else:
            features[ticker]['capex_intensity'] = np.nan
            
    return pd.DataFrame.from_dict(features, orient='index')

# Imputation done only after creating the final feature set.
def impute_final_features(feature_df, pattern_df):
    """
    Impute missing values in final feature set
    """
    df = feature_df.copy()
    
    # 1. First impute log_assets since we use it for size categories
    df['log_assets'] = df['log_assets'].fillna(df['log_assets'].median())
    
    # 2. Size-based grouping for more accurate imputation
    df['size_category'] = pd.qcut(df['log_assets'], q=5, labels=['vs', 's', 'm', 'l', 'vl'])
    
    # 3. Impute different feature groups
    # Ratio features - median within size category
    ratio_features = [col for col in df.columns if col.startswith(('avg_', 'ROA', 'ROE'))]
    for col in ratio_features:
        df[col] = df.groupby('size_category')[col].transform(
            lambda x: x.fillna(x.median())
        )
    
    # Growth features - 0 imputation with flag
    growth_features = [col for col in df.columns if 'growth' in col]
    for col in growth_features:
        df[f'{col}_imputed'] = df[col].isna().astype(int)
        df[col] = df[col].fillna(0)
    
    # Volatility features - median imputation
    vol_features = [col for col in df.columns if 'volatility' in col]
    for col in vol_features:
        df[col] = df[col].fillna(df[col].median())
        
    # Operating metrics - median within size category with flag
    operating_features = ['capex_intensity']
    for col in operating_features:
        df[f'{col}_imputed'] = df[col].isna().astype(int)
        df[col] = df.groupby('size_category')[col].transform(
            lambda x: x.fillna(x.median())
        )
    
    # 4. Remove temporary columns
    df = df.drop('size_category', axis=1)
    
    # 6. Combine with pattern features
    df = pd.concat([df, pattern_df], axis=1)
    
    # 7. Verify no missing values remain
    missing = df.isna().sum()
    if missing.any():
        print("Warning: Missing values remain in columns:", missing[missing > 0].index.tolist())
    
    return df

# Just runs it all with nice prints
def create_final_dataset(df):
    """
    Main function to create final feature set
    """
    # 1. Analyze missing patterns
    print("Analyzing missing patterns...")
    pattern_features = analyze_missing_patterns(df)
    
    # 2. Calculate raw ratios
    print("Calculating financial ratios...")
    raw_ratios = calculate_raw_ratios(df)
    
    # 3. Create base features
    print("Creating base features...")
    base_features = create_base_features(df, raw_ratios)
    
    # 4. Final imputation and combination
    print("Performing final imputation...")
    final_features = impute_final_features(base_features, pattern_features)
    
    print("\nFinal dataset shape:", final_features.shape)
    print("\nFeature groups:")
    print("Financial ratios:", len([col for col in final_features.columns if col.startswith('avg_')]))
    print("Growth metrics:", len([col for col in final_features.columns if 'growth' in col]))
    print("Missing patterns:", len([col for col in final_features.columns if col in pattern_features.columns]))
    print("Imputation flags:", len([col for col in final_features.columns if 'imputed' in col]))
    
    return final_features

if __name__ == '__main__':
    # Load the data
    df = pd.read_csv('./Data/sec_data/pivoted_financial_data_2010.csv')

    # Trim the data to required period
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].dt.year >= 2010]
    df = df[df['date'].dt.year < 2019]

    # Run the pipeline
    final_df = create_final_dataset(df)
    print("\nMissing values after processing:")
    print(final_df.isna().sum()[final_df.isna().sum() > 0])

    # Adjust datatypes and save the file 
    for col in final_df.columns:
        if col.startswith('has_'):
            final_df[col] = final_df[col].astype(int)

    final_df.to_csv('./Data/processed/financial_features_2010.csv')