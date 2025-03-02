import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from wandb_utils import init_wandb_run, log_clustering_metrics
from clustering_configs import EXPERIMENTS

def run_experiment(config):
    df = pd.read_csv('./data/processed/financial_features_2010.csv')

    X = df[config.feature_set].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    run = init_wandb_run(config)

    if config.algorithm == "kmeans":
        model = KMeans(
            n_clusters=config.n_clusters,
            random_state=config.random_seed,
            **(config.kmeans_params or {})
        )
    elif config.algorithm == "hierarchical":
        model = AgglomerativeClustering(
            n_clusters=config.n_clusters,
            **(config.hierarchical_params or {})
        )
    elif config.algorithm == "dbscan":
        model = DBSCAN(**(config.dbscan_params or {}))
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")
    
    labels = model.fit_predict(X_scaled)
    centers = getattr(model, 'cluster_centers', None) # kmeans only
    
    log_clustering_metrics(labels, X_scaled, centers)

    # end wandb run
    run.finish()

if __name__ == "__main__":
    for config in EXPERIMENTS:
        print(f"Running experiment: {config.experiment_name}")
        run_experiment(config)