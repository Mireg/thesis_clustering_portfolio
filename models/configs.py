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
        'name': 'silhouette_score',
        'goal': 'maximize'
    },
    'parameters': {
        'preprocessing': {
            'values': ['standard', 'robust'] # Scaling approaches
        },
        'algorithm': {
            'values': ['kmeans', 'hierarchical', 'dbscan']  #'dbscan' later
        },
        'n_clusters': {
            'min': 3,
            'max': 50
        },
        'variance_threshold': {
            'values': [0.01, 0.02, 0.3, 0.05, 0.01]
        },
        'correlation_threshold': {
            'values': [0.65, 0.7, 0.8, 0.9]
        },
        'use_pca': {
            'values': [True, False]
        },
        'pca_variance': {
            'values': [0.7, 0.8, 0.9, 0.95]
        },
        # KMeans specific parameters
        'kmeans_max_iter': {
            'values': [300, 500, 1000]
        },
        # Hierarchical specific parameters
        'linkage': {
            'values': ['ward', 'complete', 'average']
        },
        # DBSCAN specific parameters
        'eps': {
            'min': 0.2,
            'max': 1.5
        },
        'min_samples': {
            'distribution': 'int_uniform',
            'min': 3,
            'max': 10
        }
    }
}