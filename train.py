import pickle
import re
from pathlib import Path

import pandas as pd
import database as db
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "dataset.csv"
MODEL_DIR = PROJECT_DIR / "models"

def clean_text(text):
    """Preprocesses raw ticket text by lowercasing and removing noise."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_models():
    return {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }


def select_and_fit_model(descriptions, labels, target_name):
    class_counts = labels.value_counts()
    if len(class_counts) < 2 or class_counts.min() < 2:
        raise ValueError(f"At least two examples per {target_name} class are required for cross-validation.")

    folds = min(3, int(class_counts.min()))
    cross_validator = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    best_name = None
    best_score = float("-inf")
    best_pipeline = None

    for name, model in create_models().items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=1000)),
            ("classifier", model),
        ])
        scores = cross_val_score(pipeline, descriptions, labels, cv=cross_validator, scoring="balanced_accuracy")
        mean_score = float(scores.mean())
        print(f"{name} {target_name} cross-validation balanced accuracy: {mean_score:.4f}")
        if mean_score > best_score:
            best_name = name
            best_score = mean_score
            best_pipeline = clone(pipeline)

    best_pipeline.fit(descriptions, labels)
    print(f"Selected {best_name} for {target_name} (CV balanced accuracy: {best_score:.4f})")
    return best_pipeline.named_steps["tfidf"], best_pipeline.named_steps["classifier"]


def load_training_data():
    df = pd.read_csv(DATASET_PATH)
    required_columns = {"description", "category", "priority"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}")

    db.init_db()
    corrections = pd.DataFrame(
        db.get_corrected_tickets(),
        columns=["description", "category", "priority"],
    )
    if not corrections.empty:
        df = pd.concat([df, corrections], ignore_index=True)

    df = df.dropna(subset=list(required_columns)).copy()
    df["cleaned_description"] = df["description"].map(clean_text)
    df = df[df["cleaned_description"].ne("")]
    if df.empty:
        raise ValueError("Dataset contains no usable ticket descriptions.")
    return df


def train_and_evaluate():
    df = load_training_data()

    descriptions = df["cleaned_description"]
    category_vectorizer, category_model = select_and_fit_model(
        descriptions, df["category"], "category"
    )
    priority_vectorizer, priority_model = select_and_fit_model(
        descriptions, df["priority"], "priority"
    )

    if category_vectorizer.vocabulary_ != priority_vectorizer.vocabulary_:
        raise RuntimeError("Category and priority vectorizers unexpectedly differ.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with (MODEL_DIR / "tfidf_vectorizer.pkl").open("wb") as f:
        pickle.dump(category_vectorizer, f)
    with (MODEL_DIR / "category_model.pkl").open("wb") as f:
        pickle.dump(category_model, f)
    with (MODEL_DIR / "priority_model.pkl").open("wb") as f:
        pickle.dump(priority_model, f)

    print(f"\nModel training complete. Artifacts saved to '{MODEL_DIR}'.")

if __name__ == "__main__":
    train_and_evaluate()