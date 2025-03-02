import wandb 
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
import numpy as np
from sklearn import metrics

@dataclass
class ClusteringConfig:
    # required parameters
    experiment_name: str
    feature_set: List[str]
    algorithm: str

    # optional parameters
    random_seed: int = 42
    scaler: str = "standard"
    n_clusters: Optional[int] = None # DBSCAN does not require this

    # hyperparameters
    kmeans_params: Optional[Dict] = None
    hierarchical_params: Optional[Dict] = None
    dbscan_params: Optional[Dict] = None

    # dataclass helper to convert to dictionary
    def to_dict(self):
        return asdict(self)
    
def init_wandb_run(config: ClusteringConfig):
    run = wandb.init(
        project = "thesis_clustering_portfolio",
        config = config.to_dict(),
        name = config.experiment_name
    )
    return run

def log_clustering_metrics(labels: np.ndarray,
                           features: np.ndarray,
                           centers: Optional[np.ndarray] = None):
    
    metrics_dict = {
        "silhouette_score": metrics.silhouette_score(features, labels),
        "calinski_harabasz_score": metrics.calinski_harabasz_score(features, labels),
        "davies_bouldin_score": metrics.davies_bouldin_score(features, labels)
    }

    #WCSS
    if centers is not None:
        wcss = 0
        for i in range(len(centers)):
            cluster_points = features[labels == i]
            wcss += np.sum((cluster_points - centers[i]) ** 2)
        metrics_dict["wcss"] = wcss

    wandb.log(metrics_dict)

