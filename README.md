# Spotify Genre Segmentation 🎵🎶

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)

> **An AI/ML course project that segments and classifies Spotify music tracks into genres using machine learning.**

This project applies unsupervised and supervised machine learning techniques to analyze Spotify music data, identify genre characteristics, and build predictive models for automatic genre classification.

---

## 🎵 Project Overview

With millions of songs on Spotify, automatic genre classification helps improve music recommendations, playlist curation, and music discovery. This project explores machine learning approaches to segment music into genres.

### Key Objectives
- ✅ Analyze Spotify audio features
- ✅ Perform unsupervised clustering (genre discovery)
- ✅ Build supervised genre classification models
- ✅ Compare clustering and classification approaches
- ✅ Extract genre characteristics from audio features
- ✅ Evaluate model performance and interpretability

---

## 📊 Features

### Audio Feature Analysis
- **Acoustic Features**:
  - Tempo (BPM)
  - Energy level
  - Danceability
  - Valence (musical positivity)
  - Acousticness
  - Instrumentalness
  - Loudness
  - Speechiness

- **Track Metadata**:
  - Artist name
  - Track name
  - Release date
  - Popularity
  - Duration

### Machine Learning Techniques
- **Unsupervised Learning**:
  - K-Means Clustering
  - Hierarchical Clustering
  - DBSCAN
  - Dimensionality Reduction (PCA, t-SNE)

- **Supervised Learning**:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Support Vector Machines
  - Neural Networks

### Evaluation Methods
- Silhouette Score (clustering)
- Davies-Bouldin Index
- Classification accuracy metrics
- Cross-validation
- Feature importance analysis

---

## 🔧 Tech Stack

- **Language**: Python 3.8+
- **Notebook**: Jupyter Notebook
- **Libraries**:
  - `pandas` - Data manipulation
  - `numpy` - Numerical computing
  - `scikit-learn` - Machine learning
  - `matplotlib` - Visualization
  - `seaborn` - Statistical graphics
  - `spotipy` - Spotify API (optional)

---

## 📥 Installation

### 1. Clone Repository
```bash
git clone https://github.com/vaavgit/spotify-genre-segmentation.git
cd spotify-genre-segmentation
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter
```bash
jupyter notebook
```

Open the project notebook to begin analysis.

---

## 🎯 Usage

1. **Load Spotify Data** - Import music dataset
2. **Data Exploration** - Analyze audio features
3. **Data Preprocessing** - Normalize and scale features
4. **Unsupervised Learning** - Discover genre clusters
5. **Supervised Learning** - Train classification models
6. **Model Evaluation** - Compare performance
7. **Visualization** - Create interpretable plots

---

## 📋 Dataset Description

### Audio Features
| Feature | Range | Description |
|---------|-------|-------------|
| Acousticness | 0.0-1.0 | Acoustic vs electronic |
| Danceability | 0.0-1.0 | How danceable the track is |
| Energy | 0.0-1.0 | Intensity and activity |
| Instrumentalness | 0.0-1.0 | Presence of vocals |
| Key | 0-11 | Musical key |
| Loudness | -60-0 dB | Overall loudness |
| Mode | 0-1 | Major (1) or Minor (0) |
| Speechiness | 0.0-1.0 | Presence of spoken words |
| Tempo | 0-300+ | Beats per minute (BPM) |
| Time Signature | 3-7 | Beats per measure |
| Valence | 0.0-1.0 | Musical positivity |

---

## 📊 Analysis Workflow

### 1. Data Preparation
- Load Spotify dataset
- Handle missing values
- Normalize audio features
- Feature selection/engineering

### 2. Exploratory Analysis
- Analyze feature distributions
- Correlation analysis
- Feature importance
- Outlier detection

### 3. Unsupervised Clustering
- Determine optimal number of clusters
- Apply K-Means clustering
- Visualize clusters (PCA, t-SNE)
- Interpret cluster characteristics

### 4. Supervised Classification
- Split data into train/test
- Train multiple classifiers
- Evaluate performance
- Cross-validation
- Hyperparameter tuning

### 5. Results & Insights
- Compare model performance
- Extract genre characteristics
- Visualize decision boundaries
- Generate recommendations

---

## 📁 Project Structure

```
spotify-genre-segmentation/
├── spotify_genre_segmentation.ipynb  # Main analysis notebook
├── requirements.txt                   # Python dependencies
├── data/                              # Dataset directory
│   └── spotify_tracks.csv            # Source data
├── visualizations/                    # Generated plots
├── README.md                          # This file
└── LICENSE                            # MIT License
```

---

## 💡 Key Insights

### Finding 1: Audio Features Define Genres
- Energy and danceability strongly correlate with genre
- Valence (positivity) differs significantly across genres
- Acousticness is a key differentiator

### Finding 2: Clustering vs Classification
- Unsupervised clustering reveals natural groupings
- Supervised models achieve high accuracy
- Ensemble methods outperform single models

### Finding 3: Genre Characteristics
- Electronic: High energy, high danceability, low acousticness
- Acoustic: High acousticness, lower energy, higher valence
- Hip-Hop: High speechiness, moderate energy
- Classical: Low danceability, high instrumentalness

---

## 📈 Model Performance

### Clustering Metrics
| Method | Silhouette Score | Davies-Bouldin |
|--------|------------------|----------------|
| K-Means | - | - |
| Hierarchical | - | - |
| DBSCAN | - | - |

### Classification Metrics
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | - | - | - | - |
| Random Forest | - | - | - | - |
| Gradient Boosting | - | - | - | - |
| SVM | - | - | - | - |
| Neural Network | - | - | - | - |

*Run notebook to populate results*

---

## 🎓 Learning Outcomes

This project demonstrates:
- Unsupervised vs supervised learning
- Dimensionality reduction techniques
- Feature engineering for music data
- Model selection and evaluation
- Clustering evaluation metrics
- Classification performance metrics
- Data visualization techniques
- Music information retrieval (MIR)

---

## 🚀 Future Enhancements

- [ ] Real Spotify API integration
- [ ] Recommendation engine
- [ ] New genre discovery
- [ ] Playlist generation
- [ ] Time-series playlist analysis
- [ ] Audio feature extraction from raw audio
- [ ] Deep learning (CNN, RNN) approaches
- [ ] Web app deployment

---

## 📚 References

- [Spotify API Documentation](https://developer.spotify.com/)
- [scikit-learn Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html)
- [Music Information Retrieval (MIR)](https://en.wikipedia.org/wiki/Music_information_retrieval)
- [Audio Feature Analysis](https://www.music.mcgill.ca/)

---

## 📜 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a Pull Request

---

## 📧 Contact

For questions or feedback, open an issue on GitHub.

**Made with ❤️ by [@vaavgit](https://github.com/vaavgit)**

⭐ If you found this interesting, please star the repository!
