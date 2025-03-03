from typing import List, Dict
from wandb_utils import ClusteringConfig

def combine_feature_groups(*groups):
    """Combine multiple feature groups, removing duplicates"""
    combined_features = []
    for group in groups:
        combined_features.extend(FEATURE_GROUPS[group])
    return list(dict.fromkeys(combined_features)) 

# Features
FEATURE_GROUPS = {
    'basic': [
        'avg_ROA',
        'avg_ROE', 
        'avg_OperatingMargin',
        'avg_log_Revenue',
        'avg_log_Assets'
    ],
    'profitability': [
        'avg_ROA',
        'avg_ROE',
        'avg_OperatingMargin',
        'Revenue_growth',
        'NetIncome_growth'
    ],
    'efficiency': [
        'avg_AssetTurnover',
        'avg_CurrentRatio',
        'capex_intensity',
        'OperatingMargin_volatility'
    ]
}

sweep_config = {
    'method': 'random',  # 'grid', 'random' or 'bayes'
    'metric': {
        'name': 'silhouette_score',
        'goal': 'maximize'
    },
    'parameters': {
        'algorithm': {
            'values': ['kmeans', 'hierarchical']  # add 'dbscan' later
        },
        'n_clusters': {
            'min': 3,
            'max': 10
        },
        'feature_groups': {
            'values': [
                'basic',
                'profitability',
                'efficiency',
                'basic_and_profitability',  # combinations
                'all'
            ]
        },
        # KMeans specific parameters
        'kmeans_max_iter': {
            'values': [100, 200, 300]
        },
        # Hierarchical specific parameters
        'linkage': {
            'values': ['ward', 'complete', 'average']
        }
    }
}