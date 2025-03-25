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
            'values': ['robust'] # Scaling approaches
        },
        'algorithm': {
            'values': ['hierarchical', 'kmeans', 'dbscan']
        },
        'top_n_features': {
        'values': [5, 10, 20, 25, 30] 
        },
        'n_clusters': {
            'min': 5,
            'max': 10
        },
        'variance_threshold': {
            'values': [0.01, 0.02, 0.05]
        },
        'correlation_threshold': {
            'values': [0.75, 0.85, 0.9]
        },
        'use_pca': {
            'values': [True, False]
        },
        'pca_variance': {
            'values': [0.8, 0.9, 0.95]
        },
        # KMeans specific parameters
        'kmeans_max_iter': {
            'values': [1500]
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
            'max': 15
        }
    }
}