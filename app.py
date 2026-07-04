"""
Flask API Backend — Fake News Detection
Auto-downloads dataset and trains model on first startup if no checkpoint found.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os, sys, pickle, urllib.request, zipfile
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.dataset import clean_text

app    = Flask(__name__)
CORS(app)

tfidf_model = None
MODEL_PATH  = "outputs/checkpoints/tfidf_model.pkl"


# ─────────────────────────────────────────────
# Auto-download dataset from GitHub releases
# ─────────────────────────────────────────────
def download_and_train():
    """Download dataset and train model automatically on first run."""
    global tfidf_model

    os.makedirs("data",                    exist_ok=True)
    os.makedirs("outputs/checkpoints",     exist_ok=True)

    # Use a small public fake news dataset that's freely downloadable
    true_url = "https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv"

    # We'll use sklearn's built-in fetch or a bundled small dataset
    # Instead, generate a small training set from known patterns for demo
    print("[API] No model found. Training a demo model...")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    # Minimal training data for demo deployment
    # In production: replace with full Kaggle dataset
    demo_texts = [
        "scientists discover new treatment for cancer patients study shows",
        "government announces new policy to improve healthcare",
        "stock market rises on strong economic data report",
        "researchers publish findings in peer reviewed journal nature",
        "president signs new bill into law after senate approval",
        "university study shows exercise improves mental health outcomes",
        "SHOCKING secret they dont want you to know deep state exposed",
        "BREAKING government hiding truth about vaccines wake up sheeple",
        "URGENT share before deleted illuminati controls media fake news",
        "celebrities caught in massive scandal cover up exposed truth",
        "miracle cure doctors hate this one weird trick works instantly",
        "you wont believe what they found alien conspiracy government lies",
        "scientists confirm global warming real data shows temperature rise",
        "new renewable energy project creates thousands of jobs report",
        "local election results certified after official count completed",
        "health officials recommend annual flu vaccination for prevention",
        "FRAUD election stolen evidence proof watch before banned censored",
        "they are poisoning water supply chemtrails mind control exposed",
        "billionaire elite plan to control population secret meeting leaked",
        "mainstream media lying to you real truth hidden from public view",
    ]
    demo_labels = [0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1]

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2),
                                   stop_words='english', sublinear_tf=True)),
        ('clf',   LogisticRegression(max_iter=1000, C=1.0))
    ])
    pipeline.fit(demo_texts, demo_labels)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)

    tfidf_model = pipeline
    print("[API] Demo model trained and saved!")
    print("[API] NOTE: For full accuracy, train locally with Kaggle dataset and push model.")


def load_model():
    global tfidf_model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            tfidf_model = pickle.load(f)
        print(f"[API] Model loaded from {MODEL_PATH}")
    else:
        download_and_train()


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "name"     : "Fake News Detection API",
        "status"   : "running",
        "model"    : "TF-IDF + Logistic Regression",
        "endpoints": {
            "POST /predict": "Send { text: '...' } to get credibility score",
            "GET  /health" : "Health check"
        }
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status"      : "ok",
        "model_loaded": tfidf_model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    if tfidf_model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Send JSON with key 'text'"}), 400

    raw_text = data['text']
    if not raw_text.strip():
        return jsonify({"error": "Text is empty"}), 400

    clean        = clean_text(raw_text)
    prob_fake    = float(tfidf_model.predict_proba([clean])[0, 1])
    label        = "FAKE" if prob_fake >= 0.5 else "REAL"

    if prob_fake >= 0.75:
        credibility, message = "LOW",         "Strong signs of misinformation."
    elif prob_fake >= 0.5:
        credibility, message = "MEDIUM-LOW",  "May contain misleading info. Verify before sharing."
    elif prob_fake >= 0.25:
        credibility, message = "MEDIUM-HIGH", "Appears mostly credible. Double-check key claims."
    else:
        credibility, message = "HIGH",        "Appears credible based on language patterns."

    return jsonify({
        "label"      : label,
        "confidence" : round(prob_fake, 4),
        "credibility": credibility,
        "message"    : message,
    })


if __name__ == "__main__":
    load_model()
    port = int(os.environ.get('PORT', 5000))
    print(f"\n[API] Running on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
