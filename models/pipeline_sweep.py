from dotenv import load_dotenv
import wandb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from configs import FEATURE_GROUPS, sweep_config
from sklearn import metrics
from sklearn.decomposition import PCA


load_dotenv()
wandb.login()


def get_feature_set(group_name):
    if group_name == 'all':
        # Combine all features, remove duplicates
        all_features = []
        for features in FEATURE_GROUPS.values():
            all_features.extend(features)
        return list(dict.fromkeys(all_features))  # remove duplicates
    
    if '_and_' in group_name:
        # Combine specified groups
        groups = group_name.split('_and_')
        features = []
        for group in groups:
            features.extend(FEATURE_GROUPS[group])
        return list(dict.fromkeys(features)) 
    
    return FEATURE_GROUPS[group_name]

def train():
    # Initialize wandb with sweep config
    run = wandb.init()
    
    df = pd.read_csv('data/processed/financial_features_2010.csv')
    
    # Get features for this run
    #features = get_feature_set(run.config.feature_groups)
    features = [col for col in df.columns if col != 'Unnamed: 0' and not pd.isna(col)]

    # Log which features we're using
    wandb.log({"feature_count": len(features)})
    
    X = df[features].values

    # Scaling
    if run.config.preprocessing == 'standard':
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()

    X_scaled = scaler.fit_transform(X)
    
    # PCA
    if run.config.use_pca:
        pca = PCA(n_components=run.config.pca_variance, random_state=42)
        X_processed = pca.fit_transform(X_scaled)

        wandb.log({"pca_components_used": pca.n_components_,
                   "explained_variance": sum(pca.explained_variance_ratio_)})
    else:
        X_processed = X_scaled

    
    if run.config.algorithm == "kmeans":
        model = KMeans(
            n_clusters=run.config.n_clusters,
            max_iter=run.config.kmeans_max_iter,
            random_state=42
        )
    elif run.config.algorithm == "hierarchical":
        model = AgglomerativeClustering(
            n_clusters=run.config.n_clusters,
            linkage=run.config.linkage
        )
    elif run.config.algorithm == 'dbscan':
        model = DBSCAN(
            eps=run.config.eps,
            min_samples=run.config.min_samples
        )
    
    labels = model.fit_predict(X_processed)
    
    scores = {}

    # Calculate scores for each algorithm
    if run.config.algorithm == "dbscan":
        # check for valid clusters and noise
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2 or (len(unique_labels) == 2 and -1 in unique_labels):
            # only noise or single cluster
            scores["silhouette_score"] = -1
            scores["calinski_harabasz_score"] = -1
            scores["davies_bouldin_score"] = -1
        else:
            # Filter out noise points for silhouette
            mask = labels != -1
            if np.sum(mask) > 1:
                scores["silhouette_score"] = metrics.silhouette_score(X_processed[mask], labels[mask])
                scores["calinski_harabasz_score"] = metrics.calinski_harabasz_score(X_processed[mask], labels[mask])
                scores["davies_bouldin_score"] = metrics.davies_bouldin_score(X_processed[mask], labels[mask])
            else:
                scores["silhouette_score"] = -1
                scores["calinski_harabasz_score"] = -1
                scores["davies_bouldin_score"] = -1
                
        # Log noise percentage and number of clusters
        scores["noise_percentage"] = np.sum(labels == -1) / len(labels)
        scores["num_clusters"] = len(np.unique(labels[labels != -1]))
    else:
        # Regular metrics for KMeans and Hierarchical
        scores["silhouette_score"] = metrics.silhouette_score(X_processed, labels)
        scores["calinski_harabasz_score"] = metrics.calinski_harabasz_score(X_processed, labels)
        scores["davies_bouldin_score"] = metrics.davies_bouldin_score(X_processed, labels)
        scores["num_clusters"] = len(np.unique(labels))
    
    # For KMeans, also log inertia
    if run.config.algorithm == "kmeans":
        scores["inertia"] = model.inertia_
    
    # Log cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    cluster_distribution = dict(zip([str(u) for u in unique], counts.tolist()))
    wandb.log({"cluster_distribution": cluster_distribution})

    wandb.log(scores)

def main():
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="thesis_clustering_portfolio")
    
    # Run the sweep
    wandb.agent(sweep_id, train, count=20)  # No. of experiments

if __name__ == "__main__":
    main()