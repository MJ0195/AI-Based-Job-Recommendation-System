import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(filepath="job_recommendation_dataset.csv"):
    
    print(f"[INFO] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded successfully. Shape: {df.shape}")
    return df

def clean_text(text):
    
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = text.replace(",", " ")   # treat comma-separated skills as space-separated
    return text


def preprocess(df):
    
    print("[INFO] Cleaning text columns...")

    # Clean individual columns
    df["skills_clean"]    = df["Required Skills"].apply(clean_text)
    df["industry_clean"]  = df["Industry"].apply(clean_text)
    df["exp_clean"]       = df["Experience Level"].apply(clean_text)
    df["title_clean"]     = df["Job Title"].apply(clean_text)

    # Combine skills + industry into one text blob for richer TF-IDF matching
    df["combined_features"] = (
        df["skills_clean"] + " " +
        df["industry_clean"] + " " +
        df["title_clean"]
    )

    print("[INFO] Text cleaning complete.")
    return df

def build_tfidf(df, save_path="models/tfidf_vectorizer.pkl"):
    
    print("[INFO] Building TF-IDF matrix...")

    vectorizer = TfidfVectorizer(
        stop_words="english",   # ignore common English words
        max_features=5000,      # keep top 5000 terms to avoid noise
        ngram_range=(1, 2)      # capture single words AND two-word phrases
    )

    tfidf_matrix = vectorizer.fit_transform(df["combined_features"])
    print(f"[INFO] TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Save vectorizer for use in recommender.py and app.py
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[INFO] TF-IDF vectorizer saved to: {save_path}")

    return vectorizer, tfidf_matrix

def save_preprocessed(df, path="models/preprocessed_jobs.pkl"):
    """Save the cleaned DataFrame so other modules don't re-process."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_pickle(path)
    print(f"[INFO] Preprocessed data saved to: {path}")

if __name__ == "__main__":
    # 1. Load
    df = load_data("job_recommendation_dataset.csv")

    # 2. Clean
    df = preprocess(df)

    # 3. Vectorize
    vectorizer, tfidf_matrix = build_tfidf(df)

    # 4. Save processed data
    save_preprocessed(df)

    print("\n[SUCCESS] Preprocessing complete! Files saved in /models/")
    print("          Next step → run: python model.py")