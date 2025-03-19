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


def select_features(df, variance_threshold=0.01, correlation_threshold=0.7):
    features = df.drop(['Unnamed: 0'], axis=1, errors='ignore')
    
    variances = features.var()
    high_variance_features = variances[variances > variance_threshold].index.tolist()
    
    high_var_df = features[high_variance_features]
    corr_matrix = high_var_df.corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    
    selected_features = [col for col in high_variance_features if col not in to_drop]
    
    return selected_features

def train():
    # Initialize wandb with sweep config
    run = wandb.init()
    
    df = pd.read_csv('data/processed/financial_features_2010.csv')
    
    # Get features for this run
    selected_features = select_features(
        df,
        variance_threshold=run.config.variance_threshold,
        correlation_threshold=run.config.correlation_threshold
    )

    # Log which features we're using
    wandb.log({"feature_count": len(selected_features),
               "selected_features": selected_features})
    
    X = df[selected_features].values

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
    wandb.agent(sweep_id, train, count=100)  # No. of experiments

if __name__ == "__main__":
    main()