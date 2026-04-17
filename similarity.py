import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_similarity_scores(user_vector, tfidf_matrix):
    # cosine_similarity returns a (1, M) array; flatten to 1D
    scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    return scores


def get_top_n_jobs(df, scores, top_n=10):
    # Get indices of top N scores (descending order)
    top_indices = np.argsort(scores)[::-1][:top_n]

    # Extract the matching rows
    top_jobs = df.iloc[top_indices].copy()

    # Attach the score so callers can display it
    top_jobs["Similarity_Score"] = scores[top_indices]

    # Round for display clarity
    top_jobs["Similarity_Score"] = top_jobs["Similarity_Score"].round(4)

    return top_jobs.reset_index(drop=True)


def filter_by_experience(df, scores, experience_level=None, top_n=10):
    if experience_level:
        # Only keep rows matching the desired experience level
        mask = df["Experience Level"].str.lower() == experience_level.lower()
        filtered_df     = df[mask].copy()
        filtered_scores = scores[mask.values]
    else:
        filtered_df     = df.copy()
        filtered_scores = scores

    return get_top_n_jobs(filtered_df, filtered_scores, top_n=top_n)