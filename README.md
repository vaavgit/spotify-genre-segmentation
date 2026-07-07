# Spotify Songs' Genre Segmentation & Recommendation System

This repository contains the code, analysis, and clustering models for grouping Spotify tracks based on their auditory features and building a similarity-based recommendation engine. This is the Minor Project for the AI & ML Course.

## 🎵 Dataset Overview
The dataset contains 114,000 Spotify tracks with features representing acoustic properties:
- **Audio Features**: `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`
- **Metadata**: `track_name`, `artists`, `album_name`, `track_genre`, `popularity`

---

## 🛠️ Preprocessing and Cleaning
Data cleaning was performed in `preprocess_spotify.py` to ensure high quality and prevent redundancy:
1. **Index Removal**: Removed duplicate row indices.
2. **Missing Values**: Dropped records with missing critical metadata (e.g. empty track or artist names).
3. **Duplicate Removal**: Eliminated **24,259** duplicate tracks based on `track_id`, keeping only the first occurrence.
- **Result**: Retained **89,740** unique tracks (**78.72%** of the original data) for high-fidelity clustering.

---

## 📊 Exploratory Data Analysis & Visualizations
Visualizations were generated in `visualize_spotify.py` and saved to the `plots/` folder:
- `audio_features_distributions.png`: Shows distributions of main features like danceability, energy, valence, and tempo.
- `audio_features_correlation.png`: Correlation matrix of acoustic properties. Shows that `energy` is strongly positively correlated with `loudness` (+0.82) and strongly negatively correlated with `acousticness` (-0.73).
- `top_genres.png`: Displays track counts for the most popular genres.
- `danceability_energy_by_genre.png`: Boxplots showing how acoustic profiles (energy and danceability) vary significantly between specific genres.

---

## 🧼 Clustering and Unsupervised Learning
Using K-Means, we segmented the songs based on their scaled audio features in `cluster_spotify.py`:
1. **Elbow Method**: Plotted WCSS (Within-Cluster Sum of Squares) against $K$ (range 2 to 12) using a representative sample of 15,000 tracks. The resulting curve is saved as `plots/elbow_curve.png`.
2. **Final Model**: Selected **$K = 10$ clusters** representing distinct acoustic archetypes.
3. **PCA Visualization**: Reduced the feature space to 2D using Principal Component Analysis (PCA) and plotted the clusters. The visualization is saved as `plots/cluster_visualization.png`.

### **Cluster Characteristics**:
- **Cluster 0**: Extremely high energy, low acousticness. Top genres: *death-metal, black-metal, hardstyle*.
- **Cluster 1**: Highly acoustic, low energy/loudness. Top genres: *romance, honky-tonk, tango*.
- **Cluster 8**: Extremely low energy, very quiet ambient sounds. Top genres: *new-age, sleep, ambient*.
- **Cluster 4**: High danceability and energy (party tracks). Top genres: *dance, k-pop, turkish*.

---

## 📻 Similarity-Based Recommendation Engine
Using the clusters, we built a fast and targeted recommendation engine in `recommend_spotify.py`:
1. **Lookup**: Takes a song name (and optional artist) and finds its cluster.
2. **Filtering**: Limits search space to tracks inside the *same* cluster (acoustic profile).
3. **Similarity**: Computes the Cosine Similarity of scaled audio features between the target song and all cluster candidates.
4. **Output**: Returns the top 5 most similar tracks.

### **Example Run**:
- Input: **"Hold On"** by **Chord Overstreet** (Acoustic, Cluster 1)
- Recommendations:
  1. *Hold On* by Chord Overstreet (Similarity: 0.9997)
  2. *??* by Jay Chou (Similarity: 0.9831)
  3. *Out Of Time* by Lund (Similarity: 0.9726)
  4. *La Llamada* by Leiva (Similarity: 0.9707)
  5. *Acredita* by Novo Som (Similarity: 0.9677)

---

## 🚀 How to Run the Scripts
1. Activate your virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place the dataset `spotify_dataset.csv` inside this folder.
3. Run preprocessing:
   ```bash
   python preprocess_spotify.py
   ```
4. Run visualizations:
   ```bash
   python visualize_spotify.py
   ```
5. Run K-Means clustering:
   ```bash
   python cluster_spotify.py
   ```
6. Get song recommendations:
   ```bash
   python recommend_spotify.py
   ```
