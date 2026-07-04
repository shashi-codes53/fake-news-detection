"""
TF-IDF + Logistic Regression Model (Baseline)
Fast, lightweight, interpretable — good first baseline before BERT.
"""

import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class TFIDFModel:
    """
    TF-IDF Vectorizer + Logistic Regression pipeline.

    How it works:
    - TF-IDF converts each news article into a vector of numbers.
      Each number represents how important a word is in that article
      compared to all other articles. Rare but frequent-in-article words
      get high scores; common words like 'the' get low scores.
    - Logistic Regression then learns which word patterns
      indicate fake vs real news.
    """

    def __init__(self, max_features=50000, ngram_range=(1, 2),
                 max_iter=1000, C=1.0):
        """
        Args:
            max_features : vocab size — top N most important words
            ngram_range  : (1,2) means use single words AND word pairs
                           e.g. "not good" as one feature (bigram)
            C            : regularisation — lower = simpler model
        """
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english',      # remove 'the', 'a', 'is' etc.
                sublinear_tf=True,         # apply log to term frequencies
                strip_accents='unicode',
                analyzer='word',
                min_df=2,                  # ignore words appearing < 2 times
            )),
            ('clf', LogisticRegression(
                max_iter=max_iter,
                C=C,
                solver='lbfgs',
                n_jobs=-1,                 # use all CPU cores
            ))
        ])
        self.is_trained = False

    def train(self, texts, labels):
        """
        Train the model.
        Args:
            texts  : list of strings (news articles)
            labels : list of ints (0 = real, 1 = fake)
        """
        print("[TF-IDF Model] Training...")
        self.pipeline.fit(texts, labels)
        self.is_trained = True

        # Training accuracy
        train_preds = self.pipeline.predict(texts)
        acc = accuracy_score(labels, train_preds)
        print(f"[TF-IDF Model] Training Accuracy: {acc:.4f}")
        return acc

    def predict(self, texts):
        """
        Predict labels for a list of texts.
        Returns: numpy array of 0s and 1s
        """
        assert self.is_trained, "Train the model first!"
        return self.pipeline.predict(texts)

    def predict_proba(self, texts):
        """
        Returns probability of being fake news (class 1).
        Returns: numpy array of floats between 0 and 1
        """
        assert self.is_trained, "Train the model first!"
        probs = self.pipeline.predict_proba(texts)
        return probs[:, 1]  # probability of class 1 (fake)

    def evaluate(self, texts, labels, split_name="Test"):
        """Full evaluation with accuracy, precision, recall, F1."""
        preds = self.predict(texts)
        probs = self.predict_proba(texts)

        acc = accuracy_score(labels, preds)
        report = classification_report(labels, preds,
                                       target_names=["Real", "Fake"])
        cm = confusion_matrix(labels, preds)

        print(f"\n{'='*50}")
        print(f"  {split_name} Results — TF-IDF + LogReg")
        print(f"{'='*50}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"\n{report}")
        print(f"  Confusion Matrix:")
        print(f"  {cm}")
        print(f"{'='*50}\n")

        return {"accuracy": acc, "report": report, "confusion_matrix": cm}

    def get_top_features(self, n=20):
        """
        Show the top N words most associated with fake and real news.
        Great for understanding what the model learned.
        """
        feature_names = self.pipeline.named_steps['tfidf'].get_feature_names_out()
        coefs = self.pipeline.named_steps['clf'].coef_[0]

        top_fake = np.argsort(coefs)[-n:][::-1]
        top_real = np.argsort(coefs)[:n]

        print("\n Top words associated with FAKE news:")
        for i in top_fake:
            print(f"   {feature_names[i]:30s}  coef={coefs[i]:.3f}")

        print("\n Top words associated with REAL news:")
        for i in top_real:
            print(f"   {feature_names[i]:30s}  coef={coefs[i]:.3f}")

    def save(self, path="outputs/checkpoints/tfidf_model.pkl"):
        """Save model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        print(f"[TF-IDF Model] Saved → {path}")

    def load(self, path="outputs/checkpoints/tfidf_model.pkl"):
        """Load model from disk."""
        with open(path, 'rb') as f:
            self.pipeline = pickle.load(f)
        self.is_trained = True
        print(f"[TF-IDF Model] Loaded ← {path}")


if __name__ == "__main__":
    # Quick smoke test with dummy data
    texts  = ["Breaking: Scientists discover cure", "SHOCKING: Government hiding truth!!!",
               "New study shows vaccine effective", "FAKE MEDIA lying about election FRAUD"]
    labels = [0, 1, 0, 1]

    model = TFIDFModel()
    model.train(texts, labels)
    preds = model.predict(["Scientists publish new research"])
    probs = model.predict_proba(["Scientists publish new research"])
    print(f"Prediction: {'Fake' if preds[0] == 1 else 'Real'} (confidence: {probs[0]:.2f})")
