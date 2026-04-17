# Job Recommendation System — Project Walkthrough

## 1. Project Overview
This project is a simple AI-powered job recommender built with Streamlit. It combines two recommendation signals:

- **Skill similarity** using TF-IDF + cosine similarity
- **Job relevance prediction** using a Random Forest model

The app suggests jobs based on your entered skills and optional filters for experience, industry, and location.

---

## 2. Main Files and Responsibilities

### `app.py`
- Streamlit user interface
- Sidebar filters for experience, industry, location, and result count
- Skill input area with a quick skill picker
- Calls `recommend_jobs()` from `recommender.py`
- Displays results in card or table view
- Shows applied filters and hybrid score results

### `recommender.py`
- Loads the saved vectorizer, ML model, and preprocessed job data
- Converts user skills into a TF-IDF vector
- Computes cosine similarity between user input and every job posting
- Uses the Random Forest model to compute a relevance probability for each job
- Combines both signals into a final hybrid score
- Applies optional filters for experience, industry, and location
- Returns the top N ranked jobs

### `similarity.py`
- Computes cosine similarity between user skill vector and job vectors
- Includes a helper to select top-N jobs by score

### `data_preprocessing.py`
- Loads the raw dataset from `job_recommendation_dataset.csv`
- Cleans text fields and builds a combined text feature
- Fits a TF-IDF vectorizer and saves it to `models/tfidf_vectorizer.pkl`
- Saves the cleaned DataFrame to `models/preprocessed_jobs.pkl`

### `model.py`
- Loads preprocessed data and the TF-IDF vectorizer
- Engineers numeric features for training the ML model
- Creates a target label for `Recommended` jobs
- Trains a Random Forest classifier
- Evaluates model performance and saves the trained model to `models/rf_model.pkl`

---

## 3. Data Flow

1. `data_preprocessing.py` reads the raw CSV.
2. It cleans text fields and creates `combined_features`.
3. A TF-IDF vectorizer is trained on `combined_features`.
4. Cleaned data is saved to `models/preprocessed_jobs.pkl` and the vectorizer is saved to `models/tfidf_vectorizer.pkl`.
5. `model.py` loads these artifacts, engineer features, trains the Random Forest, and saves `models/rf_model.pkl`.
6. `app.py` loads the artifacts through `recommender.py` and serves recommendations.

---

## 4. Recommendation Logic

### Skill similarity
- User skills are cleaned, lowercased, and vectorized with the same TF-IDF vectorizer used during preprocessing.
- Cosine similarity scores are computed against each job’s `combined_features`.
- This measures how much the user’s skills and industry preference overlap with each job description.

### ML prediction
- The Random Forest model predicts a probability that a job is "recommended".
- It uses features like:
  - `Match_Score` (text similarity to average job profile)
  - `Salary_Norm` (normalized salary)
  - `Experience_Code`
  - `Industry_Code`

### Hybrid score
- Final score = `alpha * similarity + beta * ml_score`
- `alpha` comes from the sidebar slider as `Skill Similarity Weight`
- `beta = 1.0 - alpha`
- A higher alpha means the system trusts the skill match more; a lower alpha means it trusts the ML relevance model more.

---

## 5. App Behavior and Filters

### Input process
- User enters skills in a text area
- User can also select suggested skills from the quick picker
- The app merges typed skills and selection automatically

### Filters
- `Experience Level` filter: Entry, Mid, Senior, or Any
- `Preferred Industry` filter: Software, Healthcare, Finance, etc.
- `Preferred Location` filter: Sydney, San Francisco, New York, Berlin, London, Bangalore, Toronto, or Any

### Filtering logic
- All selected filters are applied together using an AND condition.
- If a filter is set to "Any" or left blank, it is ignored.
- The app only ranks jobs that satisfy the selected constraints.

---

## 6. Recent Enhancements

### Added location filtering
- `app.py` now includes a `Preferred Location` sidebar selectbox.
- `recommender.py` now accepts `location_preference` and filters results accordingly.

### Fixed sidebar dropdown visibility
- Updated CSS in `app.py` so selected dropdown text is visible and the sidebar theme stays intact.

### Clarified score weights
- Added simple explanatory text in the sidebar:
  - "Higher value means the model trusts your skill match more; lower value means it weights the job relevance model more."

### Added applied filters display
- After recommendations are generated, the app now shows exactly which filters were used.

---

## 7. How to Run the Project

1. Install requirements (if not already installed):
   ```bash
   pip install streamlit pandas scikit-learn
   ```

2. Preprocess the data:
   ```bash
   python data_preprocessing.py
   ```

3. Train the model:
   ```bash
   python model.py
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## 8. Notes for Improvement
- The job dataset is small; a larger dataset would improve recommendations.
- The ML target is heuristic-based and can be improved with real user feedback.
- Adding a training step for `Location` and `Industry` text features in the model could make recommendations more robust.
- A better UI could allow multi-select for filters and more advanced search options.
