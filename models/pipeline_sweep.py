from dotenv import load_dotenv
import wandb
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from configs import FEATURE_GROUPS, sweep_config
from sklearn import metrics

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
    features = get_feature_set(run.config.feature_groups)
    
    # Log which features we're using
    wandb.log({
        "feature_count": len(features),
        "features_used": features
    })
    
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
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
    
    labels = model.fit_predict(X_scaled)
    
    scores = {
        "silhouette_score": metrics.silhouette_score(X_scaled, labels),
        "calinski_harabasz_score": metrics.calinski_harabasz_score(X_scaled, labels),
        "davies_bouldin_score": metrics.davies_bouldin_score(X_scaled, labels)
    }
    
    wandb.log(scores)

def main():
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="thesis_clustering_portfolio")
    
    # Run the sweep
    wandb.agent(sweep_id, train, count=20)  # No. of experiments

if __name__ == "__main__":
    main()