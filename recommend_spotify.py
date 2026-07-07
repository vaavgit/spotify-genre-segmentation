import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

class SpotifyRecommender:
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.clustered_csv = os.path.join(project_dir, "spotify_clustered.csv")
        self.scaler_path = os.path.join(project_dir, "spotify_scaler.pkl")
        self.kmeans_path = os.path.join(project_dir, "spotify_kmeans_model.pkl")
        
        # Load dataset and models
        if not os.path.exists(self.clustered_csv):
            raise FileNotFoundError("Clustered dataset not found. Please run cluster_spotify.py first.")
            
        self.df = pd.read_csv(self.clustered_csv)
        self.scaler = joblib.load(self.scaler_path)
        self.kmeans = joblib.load(self.kmeans_path)
        
        self.audio_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                               'speechiness', 'acousticness', 'instrumentalness', 
                               'liveness', 'valence', 'tempo']
        
    def recommend_by_track_name(self, track_name, artist_name=None, top_n=5):
        print(f"\nSearching for track: '{track_name}'...")
        
        # Filter matches (case-insensitive)
        matches = self.df[self.df['track_name'].str.lower() == track_name.lower()]
        
        if matches.empty:
            # Fallback search: partial match
            matches = self.df[self.df['track_name'].str.lower().str.contains(track_name.lower())]
            if matches.empty:
                print(f"Track '{track_name}' not found in the dataset.")
                return None
                
        if artist_name:
            artist_matches = matches[matches['artists'].str.lower().str.contains(artist_name.lower())]
            if not artist_matches.empty:
                matches = artist_matches
                
        # If multiple matches, select the most popular one
        input_song = matches.sort_values(by='popularity', ascending=False).iloc[0]
        
        safe_track = str(input_song['track_name']).encode('ascii', 'replace').decode('ascii')
        safe_artists = str(input_song['artists']).encode('ascii', 'replace').decode('ascii')
        print(f"Found song: '{safe_track}' by {safe_artists} (Popularity: {input_song['popularity']}, Cluster: {input_song['cluster']}, Genre: {input_song['track_genre']})")
        
        # Extract features for input song and scale them
        input_features = input_song[self.audio_features].values.reshape(1, -1)
        input_scaled = self.scaler.transform(pd.DataFrame([input_song[self.audio_features]], columns=self.audio_features))
        
        # Filter candidate songs to the SAME cluster
        cluster_id = input_song['cluster']
        candidates = self.df[self.df['cluster'] == cluster_id].copy()
        
        # Exclude the input song from recommendations
        candidates = candidates[candidates['track_id'] != input_song['track_id']]
        
        # Scale candidates features
        candidates_features = candidates[self.audio_features]
        candidates_scaled = self.scaler.transform(candidates_features)
        
        # Compute cosine similarity
        similarities = cosine_similarity(input_scaled, candidates_scaled).flatten()
        candidates['similarity'] = similarities
        
        # Get top N similar songs
        recommendations = candidates.sort_values(by='similarity', ascending=False).head(top_n)
        
        print(f"\nTop {top_n} Recommendations:")
        rec_display = recommendations[['track_name', 'artists', 'track_genre', 'similarity', 'popularity']].copy()
        rec_display['track_name'] = rec_display['track_name'].apply(lambda x: str(x).encode('ascii', 'replace').decode('ascii'))
        rec_display['artists'] = rec_display['artists'].apply(lambda x: str(x).encode('ascii', 'replace').decode('ascii'))
        print(rec_display.to_string(index=False))
        
        return recommendations

def main():
    project_dir = r"C:\Users\Vaibhav\.gemini\antigravity\scratch\spotify_project"
    
    try:
        recommender = SpotifyRecommender(project_dir)
        
        # Test Recommendations
        # Test 1: Gen Hoshino - Comedy (Acoustic)
        recommender.recommend_by_track_name("Comedy", artist_name="Gen Hoshino")
        
        # Test 2: Chord Overstreet - Hold On
        recommender.recommend_by_track_name("Hold On", artist_name="Chord Overstreet")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
