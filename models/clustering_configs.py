from typing import List, Dict
from wandb_utils import ClusteringConfig

# Features
BASIC_FEATURES = [
    'avg_ROA',
    'avg_ROE', 
    'avg_OperatingMargin',
    'avg_log_Revenue',
    'avg_log_Assets'
]

PROFITABILITY_FEATURES = [
    'avg_ROA',
    'avg_ROE',
    'avg_OperatingMargin',
    'avg_log_Revenue',
    'Revenue_growth',
    'avg_log_NetIncome'
]

EFFICIENCY_FEATURES = [
    'avg_AssetTurnover',
    'avg_CurrentRatio',
    'capex_intensity',
    'avg_log_OperatingIncome',
    'OperatingMargin_volatility'
]

EXPERIMENTS = [
    ClusteringConfig(
        experiment_name="kmeans_basic_5clusters",
        feature_set=BASIC_FEATURES,
        algorithm="kmeans",
        n_clusters=5,
        kmeans_params={"max_iter": 300}
    )
]