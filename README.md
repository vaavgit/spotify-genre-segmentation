# Spotify Genre Segmentation & Recommendation System - Minor Project

This is my minor project for the AI & ML course. I built a system that groups Spotify tracks into clusters based on their audio features (acousticness, danceability, energy, etc.) and uses these clusters to make similar song recommendations.

---

## 🛠️ Data Cleaning & Preprocessing
To get clean clusters, I preprocessed the data:
1. **Removed Index Column**: Removed the duplicate row indices.
2. **Missing Metadata**: Dropped any rows where critical fields like track name or artist name were empty.
3. **Duplicates**: Spotify datasets often have duplicate tracks if they are on multiple playlists. I dropped duplicate tracks based on `track_id` (removed 24,259 tracks).
- **Final Dataset**: Kept **89,740 unique tracks** (**78.72%** of the original data).

---

## 📈 Key Insights from Data Analysis (EDA)
I generated several exploratory plots (saved in the `plots/` folder):
- `audio_features_distributions.png`: Shows the distribution of features like danceability, energy, valence, and tempo.
- `audio_features_correlation.png`: Correlation matrix of acoustic properties. Shows that `energy` is strongly positively correlated with `loudness` (+0.82) and strongly negatively correlated with `acousticness` (-0.73).
- `top_genres.png`: Displays track counts for the most popular genres.
- `danceability_energy_by_genre.png`: Boxplots showing how acoustic profiles (energy and danceability) vary significantly between specific genres.

---

## 🧼 Clustering (K-Means)
To group the songs, I scaled the audio features using `StandardScaler` and ran K-Means:
1. **Elbow Method**: Plotted WCSS (Within-Cluster Sum of Squares) against $K$ (range 2 to 12) using a representative sample of 15,000 tracks. The resulting curve is saved as `plots/elbow_curve.png`.
2. **Final Clusters**: Based on the elbow curve, I selected **K=10 clusters**. The 2D PCA projection of the clusters is saved as `plots/cluster_visualization.png`.

### **Cluster Profiles & Characteristics**:
- **Cluster 0**: High energy, low acousticness (contains *death-metal, black-metal, hardstyle*).
- **Cluster 1**: Acoustic, slow, romantic songs (contains *romance, honky-tonk, tango*).
- **Cluster 8**: Very quiet, slow ambient music (contains *new-age, sleep, ambient*).
- **Cluster 4**: Upbeat, high-energy pop and dance music (contains *dance, k-pop, turkish*).

---

## 📻 Similarity-Based Recommendation Engine
I built a similarity-based recommendation engine (`recommend_spotify.py`) using **Cosine Similarity**:
1. It searches for a user-inputted song name.
2. It finds which cluster that song belongs to.
3. It filters the dataset to only search inside that same cluster (this keeps recommendations accurate and fast).
4. It computes the cosine similarity between the scaled audio features of your song and all other songs in the cluster.
5. It returns the top 5 closest matches.

### **Example Recommendation Run**:
- Input: **"Hold On"** by **Chord Overstreet** (Acoustic, Cluster 1)
- Recommendations:
  1. *Hold On* by Chord Overstreet (Similarity: 0.9997)
  2. *??* by Jay Chou (Similarity: 0.9831)
  3. *Out Of Time* by Lund (Similarity: 0.9726)
  4. *La Llamada* by Leiva (Similarity: 0.9707)
  5. *Acredita* by Novo Som (Similarity: 0.9677)

---

## 📁 Project Structure
```text
spotify-genre-segmentation/
├── preprocess_spotify.py    # Data cleaning and feature scaling
├── visualize_spotify.py     # EDA and audio feature correlation
├── cluster_spotify.py       # K-Means clustering and evaluation
├── recommend_spotify.py     # Cosine similarity-based recommendation engine
├── requirements.txt         # Dependencies
├── README.md                # This file
├── LICENSE                  # MIT License
├── spotify_kmeans_model.pkl # Serialized K-Means model
├── spotify_scaler.pkl       # Serialized feature scaler
└── plots/                   # Saved clustering and EDA plots
```

---

## 🚀 How to Run
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Put `spotify_dataset.csv` in the same directory.
3. Run preprocessing:
   ```bash
   python preprocess_spotify.py
   ```
4. Run plots:
   ```bash
   python visualize_spotify.py
   ```
5. Run clustering:
   ```bash
   python cluster_spotify.py
   ```
6. Get recommendations:
   ```bash
   python recommend_spotify.py
   ```

---

## 📜 License
This project is open-source under the **MIT License**. See [LICENSE](LICENSE) for details.
