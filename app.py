"""
Flask API Backend — Fake News Detection
The Chrome extension sends text here and gets back a credibility score.

Endpoints:
    POST /predict   — returns label + confidence for given text
    GET  /health    — health check
    GET  /           — simple info page
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import pickle
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.dataset import clean_text

app = Flask(__name__)
CORS(app)  # Allow Chrome extension to call this API

# ─────────────────────────────────────────────
# Load model on startup
# ─────────────────────────────────────────────
tfidf_model = None

def load_model():
    global tfidf_model
    model_path = "outputs/checkpoints/tfidf_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            tfidf_model = pickle.load(f)
        print(f"[API] TF-IDF model loaded from {model_path}")
    else:
        print(f"[API] WARNING: No model found at {model_path}")
        print(f"[API] Run train.py first to generate a checkpoint.")


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "name"   : "Fake News Detection API",
        "status" : "running",
        "model"  : "TF-IDF + Logistic Regression",
        "endpoints": {
            "POST /predict": "Send { text: '...' } to get credibility score",
            "GET /health"  : "Health check"
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
    """
    Request body (JSON):
        { "text": "news article text here..." }

    Response:
        {
            "label"     : "FAKE" or "REAL",
            "confidence": 0.87,        <- probability of being fake
            "credibility": "LOW",      <- LOW / MEDIUM / HIGH
            "message"   : "..."        <- human-readable message
        }
    """
    if tfidf_model is None:
        return jsonify({"error": "Model not loaded. Run train.py first."}), 503

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Send JSON with key 'text'"}), 400

    raw_text  = data['text']
    if not raw_text.strip():
        return jsonify({"error": "Text is empty"}), 400

    clean     = clean_text(raw_text)
    prob_fake = float(tfidf_model.predict_proba([clean])[0, 1])
    label     = "FAKE" if prob_fake >= 0.5 else "REAL"

    # Credibility tiers
    if prob_fake >= 0.75:
        credibility = "LOW"
        message     = "This article shows strong signs of misinformation."
    elif prob_fake >= 0.5:
        credibility = "MEDIUM-LOW"
        message     = "This article may contain misleading information. Verify before sharing."
    elif prob_fake >= 0.25:
        credibility = "MEDIUM-HIGH"
        message     = "This article appears mostly credible but double-check key claims."
    else:
        credibility = "HIGH"
        message     = "This article appears credible based on language patterns."

    return jsonify({
        "label"      : label,
        "confidence" : round(prob_fake, 4),
        "credibility": credibility,
        "message"    : message,
        "char_count" : len(raw_text),
    })


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    print("\n[API] Starting Flask server on http://localhost:5000")
    print("[API] Press Ctrl+C to stop\n")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
