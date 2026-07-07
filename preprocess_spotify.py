import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def clean_spotify_data(df):
    initial_shape = df.shape[0]
    print(f"Initial dataset size: {initial_shape} rows")
    
    # 1. Drop the first unnamed index column if it exists
    if df.columns[0] == '' or df.columns[0].startswith('Unnamed'):
        df = df.iloc[:, 1:]
        
    # 2. Handle missing values
    # Drop rows where critical metadata (track_name, artists) is missing
    null_rows = df[df['track_name'].isna() | df['artists'].isna()].shape[0]
    if null_rows > 0:
        print(f"Dropping {null_rows} rows with missing track_name or artists...")
        df = df.dropna(subset=['track_name', 'artists'])
        
    # 3. Handle duplicates
    # A track might appear multiple times if it belongs to different genres or albums.
    # We drop duplicates based on track_id, keeping the first occurrence.
    duplicates_count = df.duplicated(subset=['track_id']).sum()
    print(f"Found {duplicates_count} duplicate track IDs. Dropping duplicates...")
    df = df.drop_duplicates(subset=['track_id'], keep='first')
    
    final_shape = df.shape[0]
    print(f"Cleaned dataset size: {final_shape} rows")
    print(f"Preserved {final_shape / initial_shape * 100:.2f}% of the original data.")
    
    return df

def main():
    project_dir = r"C:\Users\Vaibhav\.gemini\antigravity\scratch\spotify_project"
    csv_path = os.path.join(project_dir, "spotify_dataset.csv")
    cleaned_path = os.path.join(project_dir, "spotify_cleaned.csv")
    
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Clean dataset
    df_cleaned = clean_spotify_data(df)
    
    # Save cleaned data
    df_cleaned.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned Spotify dataset to {cleaned_path}")

if __name__ == "__main__":
    main()
