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
    'method': 'bayes',  # 'grid', 'random' or 'bayes'
    'metric': {
        'name': 'combined_score',
        'goal': 'maximize'
    },
    'parameters': {
        'preprocessing': {
            'values': ['standard'] # Scaling approaches
        },
        'algorithm': {
            'values': ['hierarchical']
        },
        'n_clusters': {
            'min': 3,
            'max': 50
        },
        'variance_threshold': {
            'values': [0.02]
        },
        'correlation_threshold': {
            'values': [0.8]
        },
        'use_pca': {
            'values': [True]
        },
        'pca_variance': {
            'values': [0.8, 0.9, 0.95]
        },
        # KMeans specific parameters
        'kmeans_max_iter': {
            'values': [500, 1000, 1250]
        },
        # Hierarchical specific parameters
        'linkage': {
            'values': ['ward']
        },
        # DBSCAN specific parameters
        'eps': {
            'min': 0.3,
            'max': 1.2
        },
        'min_samples': {
            'distribution': 'int_uniform',
            'min': 5,
            'max': 30
        }
    }
}