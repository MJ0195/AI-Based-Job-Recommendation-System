import pickle
import pandas as pd
import numpy as np

from similarity import get_similarity_scores, get_top_n_jobs

def load_models(
    vectorizer_path="models/tfidf_vectorizer.pkl",
    model_path="models/rf_model.pkl",
    data_path="models/preprocessed_jobs.pkl"
):
    """Load the TF-IDF vectorizer, ML model, and preprocessed DataFrame."""
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    with open(model_path, "rb") as f:
        ml_model = pickle.load(f)

    df = pd.read_pickle(data_path)

    return vectorizer, ml_model, df

def prepare_user_input(user_skills: str, industry: str = "", vectorizer=None):
    # Normalize the same way we did during preprocessing
    cleaned = user_skills.lower().replace(",", " ").strip()
    if industry:
        cleaned += " " + industry.lower().strip()

    # Transform using the ALREADY FITTED vectorizer (don't re-fit!)
    user_vector = vectorizer.transform([cleaned])
    return user_vector


FEATURE_COLS = ["Match_Score", "Salary_Norm", "Experience_Code", "Industry_Code"]

def get_ml_scores(df, ml_model):
    features = df[FEATURE_COLS]
    # predict_proba returns [[prob_class0, prob_class1], ...]
    proba = ml_model.predict_proba(features)[:, 1]
    return proba


def combine_scores(similarity_scores, ml_scores,
                   alpha=0.6, beta=0.4):
    hybrid = alpha * similarity_scores + beta * ml_scores
    return hybrid


def recommend_jobs(
    user_skills: str,
    industry_preference: str = "",
    experience_level: str = "",
    location_preference: str = "",
    top_n: int = 10,
    alpha: float = 0.6,
    beta: float = 0.4
):
    # 1. Load artifacts
    vectorizer, ml_model, df = load_models()

    # 2. Vectorize user input
    user_vector = prepare_user_input(user_skills, industry_preference, vectorizer)

    # 3. Cosine similarity vs. all jobs
    tfidf_matrix = vectorizer.transform(df["combined_features"])
    sim_scores   = get_similarity_scores(user_vector, tfidf_matrix)

    # 4. ML probability scores
    ml_proba = get_ml_scores(df, ml_model)

    # 5. Hybrid score
    hybrid = combine_scores(sim_scores, ml_proba, alpha, beta)

    # 6. Apply filters
    mask = pd.Series([True] * len(df), index=df.index)
    
    if experience_level and experience_level.strip():
        mask &= df["Experience Level"].str.lower() == experience_level.lower()
    
    if industry_preference and industry_preference.strip():
        mask &= df["Industry"].str.lower() == industry_preference.lower()
    
    if location_preference and location_preference.strip():
        mask &= df["Location"].str.lower() == location_preference.lower()
    
    df_filt = df[mask].copy()
    hybrid_filt = hybrid[mask.values]
    sim_filt = sim_scores[mask.values]

    # 7. Rank by hybrid score
    top_idx = np.argsort(hybrid_filt)[::-1][:top_n]
    results = df_filt.iloc[top_idx].copy()

    # 8. Attach display scores
    results["Similarity_Score"] = np.round(sim_filt[top_idx], 4)
    results["Hybrid_Score"]     = np.round(hybrid_filt[top_idx], 4)

    # 9. Select and rename columns for display
    output_cols = [
        "Job Title", "Company", "Location",
        "Experience Level", "Salary", "Industry",
        "Required Skills", "Similarity_Score", "Hybrid_Score"
    ]
    results = results[output_cols].reset_index(drop=True)
    results.index += 1   # 1-based ranking for display

    return results

if __name__ == "__main__":
    print("\n🔍  Job Recommendation System — Quick Test\n")
    user_input = input("Enter your skills (comma-separated): ").strip()
    industry   = input("Preferred industry (press Enter to skip): ").strip()
    exp_level  = input("Experience level (Entry/Mid/Senior — or Enter to skip): ").strip()
    location   = input("Preferred location (press Enter to skip): ").strip()

    results = recommend_jobs(
        user_skills=user_input,
        industry_preference=industry,
        experience_level=exp_level,
        location_preference=location,
        top_n=10
    )

    print(f"\n{'='*70}")
    print(f"  TOP {len(results)} RECOMMENDED JOBS")
    print(f"{'='*70}")
    print(results.to_string())