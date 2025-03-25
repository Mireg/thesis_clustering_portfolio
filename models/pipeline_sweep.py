import wandb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn import metrics
from dotenv import load_dotenv
from configs import FEATURE_GROUPS, sweep_config

# Load env variables and login
load_dotenv()
wandb.login()

def select_features(df, variance_threshold=0.01, correlation_threshold=0.7, top_n=None):
    ticker_col = None
    if 'Unnamed: 0' in df.columns:
        ticker_col = 'Unnamed: 0'
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    features = df[numeric_cols]
    
    if ticker_col and ticker_col in features.columns:
        features = features.drop(columns=[ticker_col])
    
    # Feature selection - variance
    variances = features.var()
    high_variance = variances[variances > variance_threshold].index.tolist()
    # Feature selection - correlation
    corr_matrix = df[high_variance].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > correlation_threshold)]
    
    selected = [col for col in high_variance if col not in to_drop]
    return selected[:top_n] if top_n else selected

def calculate_wcss_bcss(X, labels):
    valid_mask = labels != -1
    if np.sum(valid_mask) == 0:
        return 0.0, 0.0
    
    clustered_X = X[valid_mask]
    clustered_labels = labels[valid_mask]
    
    unique_labels, counts = np.unique(clustered_labels, return_counts=True)
    if len(unique_labels) < 1:
        return 0.0, 0.0
    
    centroids = np.array([clustered_X[clustered_labels == l].mean(0) for l in unique_labels])
    global_centroid = clustered_X.mean(0)
    
    # WCSS calculation
    diffs = clustered_X - centroids[clustered_labels]
    wcss = np.sum(np.square(diffs))
    
    # BCSS calculation
    centroid_diffs = centroids - global_centroid
    bcss = np.sum(counts * np.sum(np.square(centroid_diffs), axis=1))
    
    return wcss, bcss

def calculate_combined_score(metrics_dict):
    # Metrics normalization to avoid domination from one metric
    silhouette = metrics_dict.get('silhouette', 0)  # Range: [-1, 1] 
    # Rescaling silhouette from [-1,1] to [0,1]
    silhouette = (silhouette + 1) / 2
    
    # Log-transform Calinski-Harabasz for better scaling with outliers
    # It can take any value and can easily dominate other metrics
    calinski = metrics_dict.get('calinski', 0)
    calinski_normalized = np.log1p(calinski) / np.log1p(5000) 
    calinski_normalized = min(1.0, calinski_normalized)
    
    # Conversion of Davies-Bouldin (lower is better) to [0, 1] scale
    db_inverted = 1 / (1 + metrics_dict.get('db_score', 10))
     
    wcss = metrics_dict.get('wcss', 0)
    bcss = metrics_dict.get('bcss', 0)

    # Separation ratio normalized to [0,1]
    separation_ratio = bcss / (wcss + bcss + 1e-8)  # epsilon avoids division by zero, unlikely but good practice
    separation_ratio = min(1.0, separation_ratio)
    
    return (0.35 * silhouette + 
            0.25 * calinski_normalized + 
            0.20 * db_inverted + 
            0.20 * separation_ratio)

def calculate_metrics(X, labels):
    metrics_dict = {}
    wcss, bcss = calculate_wcss_bcss(X, labels)
    metrics_dict.update({
        'wcss': wcss,
        'bcss': bcss,
        'explained_variance': bcss/(wcss + bcss) if (wcss + bcss) != 0 else 0
    })
    
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        mask = labels != -1
        if np.sum(mask) > 1:
            metrics_dict.update({
                'silhouette': metrics.silhouette_score(X[mask], labels[mask]),
                'calinski': metrics.calinski_harabasz_score(X[mask], labels[mask]),
                'db_score': metrics.davies_bouldin_score(X[mask], labels[mask]),
                'num_clusters': len(unique_labels) - (1 if -1 in unique_labels else 0),
                'noise_pct': np.mean(labels == -1)*100
            })
    
    metrics_dict['combined_score'] = calculate_combined_score(metrics_dict)
    return metrics_dict

def log_cluster_info(df, features, labels):
    df_out = df.copy()
    df_out['cluster'] = labels
    
    # Log cluster sizes
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    cluster_sizes = {f"cluster_{k}": v for k, v in cluster_counts.items()}
    wandb.log({"cluster_sizes": cluster_sizes})
    
    # Log cluster assignments
    assignments_table = wandb.Table(dataframe=df_out[['ticker', 'cluster']])
    wandb.log({"cluster_assignments": assignments_table})
    
    # Calculate feature importance as z-scores
    feature_importance = {}
    for cluster in np.unique(labels):
        if cluster == -1:  # Skip noise points
            continue
            
        mask = labels == cluster
        cluster_data = df_out[mask]
        
        for feature in features:
            # Calculate z-score
            feature_mean = cluster_data[feature].mean()
            overall_mean = df_out[feature].mean()
            overall_std = df_out[feature].std()
            
            if overall_std > 0:
                z_score = (feature_mean - overall_mean) / overall_std
                feature_importance[f"cluster_{cluster}_{feature}"] = z_score
    
    wandb.log({"feature_importance": feature_importance})
    
    profiles = []
    for cluster in np.unique(labels):
        if cluster == -1:  # Skip noise
            continue
            
        mask = labels == cluster
        cluster_data = df_out[mask]
        
        # Mean stats to understand cluster characteristics
        cluster_stats = {feature: cluster_data[feature].mean() for feature in features}
        
        # Log cluster info
        cluster_stats.update({
            "cluster": int(cluster),
            "size": int(np.sum(mask)),
            "percentage": float(np.sum(mask) / len(labels) * 100)
        })
        
        profiles.append(cluster_stats)
    
    # Log as table
    profile_df = pd.DataFrame(profiles).set_index('cluster')
    profile_table = wandb.Table(dataframe=profile_df.reset_index())
    wandb.log({"cluster_profile": profile_table})

def train():
    run = wandb.init()

    # Tagging for the run
    algorithm = run.config['algorithm']
    n_clusters = run.config.get('n_clusters', 'auto')
    feature_count = len(features)
    wandb.run.tags = [
        algorithm, 
        f"clusters_{n_clusters}",
        f"features_{feature_count}",
        "pca" if run.config.get('use_pca', False) else "no_pca"
    ]
    
    df = pd.read_csv('data/processed/financial_features_2010.csv')
    
    # Save tickers for later
    tickers = df['Unnamed: 0'].copy()
    df_with_tickers = pd.DataFrame({'ticker': tickers})
    
    # Select features
    feature_params = {
        'variance_threshold': run.config.get('variance_threshold', 0.01),
        'correlation_threshold': run.config.get('correlation_threshold', 0.7),
        'top_n': run.config.get('top_n_features', None)
    }
    
    features = select_features(df, **feature_params)
    X = df[features].values
    
    # Scaling
    scaler = RobustScaler() if run.config.get('preprocessing', 'standard') == 'robust' else StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA if configured
    if run.config.get('use_pca', False):
        pca_variance = run.config.get('pca_variance', 0.95)
        pca = PCA(n_components=pca_variance, random_state=42)
        X_scaled = pca.fit_transform(X_scaled)
        wandb.log({
            "pca_components": pca.n_components_,
            "pca_explained_variance": pca.explained_variance_ratio_.sum()
        })
    
    # Clustering
    if run.config['algorithm'] == "kmeans":
        model = KMeans(
            n_clusters=run.config['n_clusters'],
            max_iter=run.config['kmeans_max_iter'],
            random_state=42
        )
    elif run.config['algorithm'] == "hierarchical":
        model = AgglomerativeClustering(
            n_clusters=run.config['n_clusters'],
            linkage=run.config['linkage']
        )
    elif run.config['algorithm'] == "dbscan":
        model = DBSCAN(
            eps=run.config['eps'],
            min_samples=run.config['min_samples']
        )
    else:
        raise ValueError(f"Unknown algorithm: {run.config['algorithm']}")
    
    labels = model.fit_predict(X_scaled)
    
    metrics_dict = calculate_metrics(X_scaled, labels)
    
    wandb.log(metrics_dict)
    wandb.log({"feature_count": len(features)})
    wandb.log({"features": features})
    
    log_cluster_info(df_with_tickers, features, labels)
    
    return metrics_dict['combined_score']


if __name__ == "__main__":
    wandb.agent(wandb.sweep(sweep_config), train, count=50)