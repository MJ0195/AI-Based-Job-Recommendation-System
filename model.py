import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

def load_artifacts(data_path="models/preprocessed_jobs.pkl",
                   vectorizer_path="models/tfidf_vectorizer.pkl"):
    """Load the cleaned DataFrame and the fitted TF-IDF vectorizer."""
    print("[INFO] Loading preprocessed data and TF-IDF vectorizer...")
    df         = pd.read_pickle(data_path)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return df, vectorizer

def engineer_features(df, vectorizer):
    """
    Create numeric features for the ML model:

    1. Match_Score      : cosine similarity of each job's skills vs. the
                          dataset's average skill profile (proxy for popularity)
    2. Salary_Norm      : min-max normalized salary
    3. Experience_Code  : ordinal encode experience level
    4. Industry_Code    : label encode industry category
    """
    print("[INFO] Engineering features...")

    # --- Feature 1: Match_Score (cosine similarity to mean vector) ---
    tfidf_matrix = vectorizer.transform(df["combined_features"])
    mean_vector  = np.asarray(tfidf_matrix.mean(axis=0))      # average job profile (convert from np.matrix)
    scores       = cosine_similarity(tfidf_matrix, mean_vector).flatten()
    df["Match_Score"] = scores

    # --- Feature 2: Salary (normalized 0–1) ---
    sal_min = df["Salary"].min()
    sal_max = df["Salary"].max()
    df["Salary_Norm"] = (df["Salary"] - sal_min) / (sal_max - sal_min)

    # --- Feature 3: Experience level → ordinal number ---
    exp_map = {"Entry Level": 0, "Mid Level": 1, "Senior Level": 2}
    df["Experience_Code"] = df["Experience Level"].map(exp_map).fillna(1)

    # --- Feature 4: Industry → integer label ---
    le_industry = LabelEncoder()
    df["Industry_Code"] = le_industry.fit_transform(df["Industry"])

    # Save the label encoder so app.py can reuse it
    os.makedirs("models", exist_ok=True)
    with open("models/label_encoder_industry.pkl", "wb") as f:
        pickle.dump(le_industry, f)

    print("[INFO] Feature engineering complete.")
    return df

def create_target(df):
    """
    Define 'Recommended' label:
      1  → Recommended  (Match_Score > median  AND  Salary > median)
      0  → Not Recommended

    This is a reasonable heuristic: a job is 'good' when it's
    both skill-relevant and well-paying relative to the dataset.
    """
    score_threshold  = df["Match_Score"].median()
    salary_threshold = df["Salary"].median()

    df["Recommended"] = (
        (df["Match_Score"] >= score_threshold) &
        (df["Salary"]      >= salary_threshold)
    ).astype(int)

    pos = df["Recommended"].sum()
    neg = len(df) - pos
    print(f"[INFO] Target created → Recommended: {pos}, Not Recommended: {neg}")
    return df

FEATURE_COLS = ["Match_Score", "Salary_Norm", "Experience_Code", "Industry_Code"]

def train_model(df):
    """
    Train a Random Forest classifier and evaluate it.

    Random Forest:
    - Builds many decision trees and combines their predictions
    - Robust to outliers and does not require feature scaling
    - Works well on tabular data with mixed feature types
    """
    X = df[FEATURE_COLS]
    y = df["Recommended"]

    # 80/20 train-test split; stratify keeps class balance equal in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # Instantiate and train
    model = RandomForestClassifier(
        n_estimators=100,   # 100 trees gives a good accuracy/speed trade-off
        max_depth=8,        # limit depth to avoid over-fitting
        random_state=42,
        n_jobs=-1           # use all CPU cores
    )
    model.fit(X_train, y_train)
    print("[INFO] Model training complete.")

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """
    Print a detailed evaluation of the trained model:
    - Accuracy  : fraction of correct predictions
    - Confusion Matrix  : actual vs predicted class counts
    - Classification Report : precision, recall, F1-score per class

    Precision : of all predicted 'Recommended', how many were truly so?
    Recall    : of all truly 'Recommended', how many did we catch?
    F1-score  : harmonic mean of precision & recall (balanced metric)
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    cr  = classification_report(y_test, y_pred,
                                target_names=["Not Recommended", "Recommended"])

    print("\n" + "=" * 55)
    print("          MODEL EVALUATION RESULTS")
    print("=" * 55)
    print(f"\n  Accuracy Score : {acc * 100:.2f}%")
    print("\n  Confusion Matrix:")
    print(f"  {'':20s}  Pred: 0   Pred: 1")
    print(f"  {'Actual: 0 (Not Rec)':22s}  {cm[0][0]:6d}   {cm[0][1]:6d}")
    print(f"  {'Actual: 1 (Rec)':22s}  {cm[1][0]:6d}   {cm[1][1]:6d}")
    print("\n  Classification Report:")
    for line in cr.split("\n"):
        print(f"    {line}")
    print("=" * 55)

    # Interpretation hint
    print("\n  [HINT] Diagonal cells in the confusion matrix = correct predictions.")
    print("         Off-diagonal cells = mistakes (false positives / negatives).")
    print("         Aim for high F1-score on the 'Recommended' class.\n")

    return acc

def save_model(model, path="models/rf_model.pkl"):
    """Persist the trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[INFO] Model saved to: {path}")

if __name__ == "__main__":
    # 1. Load data & vectorizer
    df, vectorizer = load_artifacts()

    # 2. Feature engineering
    df = engineer_features(df, vectorizer)

    # 3. Create target label
    df = create_target(df)

    # 4. Train
    model, X_test, y_test = train_model(df)

    # 5. Evaluate
    evaluate_model(model, X_test, y_test)

    # 6. Save model
    save_model(model)

    # 7. Save updated DataFrame (now has Match_Score, Recommended, etc.)
    df.to_pickle("models/preprocessed_jobs.pkl")
    print("\n[SUCCESS] Model training complete!")
    print("          Next step → run: streamlit run app.py")