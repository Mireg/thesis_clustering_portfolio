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

def determine_optimal_k(X, max_k=20):
    results = []
    
    sse = []
    silhouette_avg = []
    ch_scores = []
    
    k_range = range(2, max_k+1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        
        sse.append(kmeans.inertia_)
        silhouette_avg.append(metrics.silhouette_score(X, labels))
        ch_scores.append(metrics.calinski_harabasz_score(X, labels))
        results.append([k, kmeans.inertia_, metrics.silhouette_score(X, labels), 
                     metrics.calinski_harabasz_score(X, labels)])
    
    table = wandb.Table(columns=["k", "sse", "silhouette", "calinski_harabasz"], 
                       data=results)
    
    wandb.log({
        "optimal_k_analysis": table,
        "optimal_k_plot": wandb.plot.line(
            table, "k", "calinski_harabasz", title="Calinski-Harabasz Score by K")
    })
    
    # Find optimal K using CH scores
    ch_k = k_range[np.argmax(ch_scores)]
    return ch_k

def train():
    run = wandb.init()
    
    df = pd.read_csv('data/processed/financial_features_2010.csv')
    
    # Get features for this run
    selected_features = select_features(
        df,
        variance_threshold=run.config.variance_threshold,
        correlation_threshold=run.config.correlation_threshold
    )

    # Log selected features with their variance
    wandb.log({
        "feature_count": len(selected_features),
        "feature_list": wandb.Table(columns=["Feature", "Variance"], 
                                data=[[f, float(df[f].var())] for f in selected_features])
    })
    
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

        wandb.log({
            "pca_components_used": pca.n_components_,
            "explained_variance": sum(pca.explained_variance_ratio_)
        })
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
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2 or (len(unique_labels) == 2 and -1 in unique_labels):
            scores["silhouette_score"] = -1
            scores["calinski_harabasz_score"] = -1
            scores["davies_bouldin_score"] = -1
        else:
            mask = labels != -1
            if np.sum(mask) > 1:
                scores["silhouette_score"] = metrics.silhouette_score(X_processed[mask], labels[mask])
                scores["calinski_harabasz_score"] = metrics.calinski_harabasz_score(X_processed[mask], labels[mask])
                scores["davies_bouldin_score"] = metrics.davies_bouldin_score(X_processed[mask], labels[mask])
            else:
                scores["silhouette_score"] = -1
                scores["calinski_harabasz_score"] = -1
                scores["davies_bouldin_score"] = -1
                
        scores["noise_percentage"] = np.sum(labels == -1) / len(labels)
        scores["num_clusters"] = len(np.unique(labels[labels != -1]))
    else:
        scores["silhouette_score"] = metrics.silhouette_score(X_processed, labels)
        scores["calinski_harabasz_score"] = metrics.calinski_harabasz_score(X_processed, labels)
        scores["davies_bouldin_score"] = metrics.davies_bouldin_score(X_processed, labels)
        scores["num_clusters"] = len(np.unique(labels))
    
    # For KMeans, log inertia and feature importance
    if run.config.algorithm == "kmeans":
        scores["inertia"] = model.inertia_
        
        centers = model.cluster_centers_
        feature_importance = np.var(centers, axis=0)
        sorted_idx = np.argsort(-feature_importance)[:10]
        
        importance_table = wandb.Table(
            columns=["Feature", "Importance"],
            data=[[selected_features[i], float(feature_importance[i])] for i in sorted_idx]
        )
        
        wandb.log({"top_features": importance_table})

    if scores["silhouette_score"] > 0:
        # Normalize DB score (lower is better)
        db_normalized = 1 / (1 + scores["davies_bouldin_score"])
        
        # Create combined score - weighted average
        combined_score = (
            0.4 * scores["silhouette_score"] + 
            0.4 * (scores["calinski_harabasz_score"] / 1000) +  # Scale down CH score
            0.2 * db_normalized
        )
        
        scores["combined_score"] = combined_score
    
    # Log cluster distribution as a table instead of dictionary
    unique, counts = np.unique(labels, return_counts=True)
    
    # Create a proper table visualization
    cluster_table = wandb.Table(columns=["Cluster", "Count", "Percentage"])
    total = len(labels)
    for cluster, count in zip(unique, counts):
        cluster_name = "Noise" if cluster == -1 else f"Cluster {cluster}"
        percentage = (count/total) * 100
        cluster_table.add_data(cluster_name, int(count), float(percentage))
    
    wandb.log({
        "cluster_distribution_table": cluster_table,
        "cluster_count": len(unique)
    })
    
    # Store original dictionary in summary only
    cluster_distribution = dict(zip([str(u) for u in unique], counts.tolist()))
    wandb.run.summary["cluster_sizes"] = cluster_distribution

    wandb.log(scores)

def main():
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="thesis_clustering_portfolio")
    
    # Run the sweep
    wandb.agent(sweep_id, train, count=50)  # No. of experiments

if __name__ == "__main__":
    main()