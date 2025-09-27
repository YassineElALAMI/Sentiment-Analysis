
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_models(input_path):
    # Load data
    df = pd.read_csv(input_path)
    X = df["clean_text"]
    y = df["label"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
    }

    for name, model in models.items():
        print(f"\n[INFO] Training {name}...")
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, preds)
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))

if __name__ == "__main__":
    train_models("data/processed/cleaned.csv")
