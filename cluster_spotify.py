import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib

def main():
    project_dir = r"C:\Users\Vaibhav\.gemini\antigravity\scratch\spotify_project"
    cleaned_csv = os.path.join(project_dir, "spotify_cleaned.csv")
    plots_dir = os.path.join(project_dir, "plots")
    
    if not os.path.exists(cleaned_csv):
        print(f"Cleaned dataset not found. Please run preprocess_spotify.py first.")
        return
        
    df = pd.read_csv(cleaned_csv)
    
    # 1. Select audio features for clustering
    audio_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                      'speechiness', 'acousticness', 'instrumentalness', 
                      'liveness', 'valence', 'tempo']
    
    X = df[audio_features]
    
    # 2. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler for the recommendation engine
    joblib.dump(scaler, os.path.join(project_dir, 'spotify_scaler.pkl'))
    
    # Use a random sample of 15k rows for the elbow method to speed things up
    print("Calculating Elbow Method curve on a representative sample...")
    sample_size = min(15000, X_scaled.shape[0])
    sample_indices = np.random.choice(X_scaled.shape[0], size=sample_size, replace=False)
    X_sample = X_scaled[sample_indices]
    
    wcss = []
    k_range = range(2, 13)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_sample)
        wcss.append(kmeans.inertia_)
        
    # Plot Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS (Inertia)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'elbow_curve.png'), dpi=300)
    plt.close()
    print(f"Elbow curve saved to {os.path.join(plots_dir, 'elbow_curve.png')}")
    
    # K=10 clusters works well as it represents distinct styles of music in the data
    optimal_k = 10
    print(f"\nTraining final K-Means with K={optimal_k} clusters on the full dataset...")
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['cluster'] = kmeans_final.fit_predict(X_scaled)
    
    # Save the clustering model
    joblib.dump(kmeans_final, os.path.join(project_dir, 'spotify_kmeans_model.pkl'))
    
    # Save clustered dataset
    clustered_csv = os.path.join(project_dir, "spotify_clustered.csv")
    df.to_csv(clustered_csv, index=False)
    print(f"Saved clustered dataset to {clustered_csv}")
    
    # 5. Dimensionality Reduction using PCA for Visualization
    print("\nPerforming PCA for cluster visualization...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df['pca1'] = X_pca[:, 0]
    df['pca2'] = X_pca[:, 1]
    
    # Plot PCA clusters (using a sample of 5,000 to keep the plot from being too cluttered)
    plt.figure(figsize=(10, 8))
    df_sample = df.sample(n=5000, random_state=42)
    sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=df_sample, 
                    palette='tab10', alpha=0.7, style='cluster')
    plt.title('Spotify Song Clusters Visualized in 2D (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'cluster_visualization.png'), dpi=300)
    plt.close()
    print(f"Cluster visualization plot saved to {os.path.join(plots_dir, 'cluster_visualization.png')}")
    
    # 6. Analyze Clusters (Characterization and Top Genres)
    print("\nCluster Analysis:")
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id]
        size = cluster_df.shape[0]
        
        # Calculate mean features for the cluster
        mean_dance = cluster_df['danceability'].mean()
        mean_energy = cluster_df['energy'].mean()
        mean_acoustic = cluster_df['acousticness'].mean()
        
        # Find top 3 genres in this cluster
        top_genres = cluster_df['track_genre'].value_counts().head(3).index.tolist()
        
        print(f"\nCluster {cluster_id}: {size} songs")
        print(f"  Averages -> Danceability: {mean_dance:.2f}, Energy: {mean_energy:.2f}, Acousticness: {mean_acoustic:.2f}")
        print(f"  Top Genres: {', '.join(top_genres)}")

if __name__ == "__main__":
    main()
