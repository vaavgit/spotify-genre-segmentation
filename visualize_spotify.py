import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for premium visualizations
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Premium Color Palettes
PALETTE_CREST = "crest"
PALETTE_FLARE = "flare"
PALETTE_VIRIDIS = "viridis"

def generate_spotify_plots(df, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Distribution of Key Audio Features
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    features_to_plot = ['danceability', 'energy', 'valence', 'tempo']
    colors = ["#4A90E2", "#E25A84", "#6A4C93", "#1A936F"]
    
    for idx, feature in enumerate(features_to_plot):
        row = idx // 2
        col = idx % 2
        sns.histplot(df[feature], kde=True, ax=axes[row, col], color=colors[idx], bins=30)
        axes[row, col].set_title(f'Distribution of {feature.title()}')
        axes[row, col].set_xlabel(feature.title())
        axes[row, col].set_ylabel('Count')
        
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'audio_features_distributions.png'), dpi=300)
    plt.close()
    
    # 2. Audio Features Correlation Heatmap
    plt.figure(figsize=(10, 8))
    audio_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                      'speechiness', 'acousticness', 'instrumentalness', 
                      'liveness', 'valence', 'tempo']
    corr = df[audio_features].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=0.5)
    plt.title('Spotify Audio Features Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'audio_features_correlation.png'), dpi=300)
    plt.close()
    
    # 3. Top 10 Genres in Dataset
    plt.figure(figsize=(12, 6))
    top_genres = df['track_genre'].value_counts().head(10)
    sns.barplot(x=top_genres.values, y=top_genres.index, hue=top_genres.index, palette="viridis", legend=False)
    plt.title('Top 10 Track Genres in Cleaned Dataset')
    plt.xlabel('Track Count')
    plt.ylabel('Genre')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'top_genres.png'), dpi=300)
    plt.close()
    
    # 4. Danceability and Energy by Genre (Top 5 Genres)
    top_5_genres = df['track_genre'].value_counts().head(5).index
    df_top_5 = df[df['track_genre'].isin(top_5_genres)]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.boxplot(x='track_genre', y='danceability', data=df_top_5, palette="pastel", ax=axes[0])
    axes[0].set_title('Danceability Distribution for Top 5 Genres')
    axes[0].set_xlabel('Genre')
    axes[0].set_ylabel('Danceability')
    
    sns.boxplot(x='track_genre', y='energy', data=df_top_5, palette="pastel", ax=axes[1])
    axes[1].set_title('Energy Distribution for Top 5 Genres')
    axes[1].set_xlabel('Genre')
    axes[1].set_ylabel('Energy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'danceability_energy_by_genre.png'), dpi=300)
    plt.close()
    
    print(f"All visualizations saved successfully in {plots_dir}")

def main():
    project_dir = r"C:\Users\Vaibhav\.gemini\antigravity\scratch\spotify_project"
    cleaned_csv = os.path.join(project_dir, "spotify_cleaned.csv")
    plots_dir = os.path.join(project_dir, "plots")
    
    if not os.path.exists(cleaned_csv):
        print(f"Cleaned dataset not found at {cleaned_csv}. Please run preprocess_spotify.py first.")
        return
        
    df = pd.read_csv(cleaned_csv)
    generate_spotify_plots(df, plots_dir)

if __name__ == "__main__":
    main()
