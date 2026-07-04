"""
Flask API — Fake News Detection
Trains model on startup if not found. Guaranteed to work on Render.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os, sys, pickle
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.dataset import clean_text

app        = Flask(__name__)
CORS(app)
MODEL_PATH = "outputs/checkpoints/tfidf_model.pkl"
tfidf_pipeline = None


def train_and_save():
    """Train a model from scratch and save it. Called on startup."""
    print("[STARTUP] Training model from scratch...")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    texts = [
        # REAL news (label 0)
        "scientists discover new treatment for cancer patients study shows promising results",
        "government announces new policy to improve public healthcare system nationwide",
        "stock market rises on strong economic data quarterly earnings report published",
        "researchers publish peer reviewed findings in nature journal on climate change",
        "president signs new infrastructure bill into law after bipartisan senate approval",
        "university study shows regular exercise significantly improves mental health outcomes",
        "new renewable energy project creates thousands of green jobs across the country",
        "health officials recommend annual flu vaccination for prevention during winter season",
        "local election results officially certified after complete recount by authorities",
        "supreme court issues ruling on landmark civil rights case affecting millions",
        "scientists confirm global temperatures rising due to greenhouse gas emissions data",
        "international trade agreement signed between multiple countries to boost economy",
        "hospital reports breakthrough surgery technique reduces recovery time significantly",
        "city council approves new public transportation project to reduce traffic congestion",
        "technology company releases new product after years of research and development",
        "study finds mediterranean diet linked to lower risk of heart disease in adults",
        "nasa successfully launches new satellite to monitor earth climate patterns",
        "federal reserve adjusts interest rates in response to inflation economic data",
        "new education policy aims to improve literacy rates in underprivileged communities",
        "world health organization updates guidelines on nutrition and physical activity",

        # FAKE news (label 1)
        "SHOCKING secret they dont want you to know deep state exposed wake up",
        "BREAKING government hiding truth about vaccines dangerous side effects covered up",
        "URGENT share before deleted illuminati controls media everything is fake news lies",
        "celebrities caught in massive scandal pedophile ring cover up exposed truth finally",
        "miracle cure doctors hate this one weird trick cures cancer instantly at home",
        "you wont believe what they found alien bodies government hiding area 51 proof",
        "FRAUD election stolen millions of votes evidence proof watch before banned censored",
        "they are poisoning water supply chemtrails mind control exposed microchips vaccines",
        "billionaire elite plan to control world population secret meeting leaked documents",
        "mainstream media lying to you real truth hidden from public view wake up sheeple",
        "BOMBSHELL whistleblower exposes government plot to microchip citizens forced vaccines",
        "new world order plan revealed globalist agenda to enslave humanity exposed finally",
        "crisis actor caught on camera fake shooting staged by government gun control agenda",
        "5g towers causing coronavirus spread truth they dont want you to know share now",
        "george soros paying protesters to destroy america open borders globalist puppet master",
        "moon landing was faked nasa admits in secret documents leaked by insider whistleblower",
        "fluoride in water lowers iq government deliberately poisoning citizens since 1950s",
        "deep state planning false flag attack to start world war three imminent warning",
        "secret cure for all diseases suppressed by big pharma to keep patients paying forever",
        "reptilian shapeshifters control world governments exposed in leaked footage share now",
    ]
    labels = [0]*20 + [1]*20

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            sublinear_tf=True,
            min_df=1,
        )),
        ('clf', LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs'))
    ])
    pipeline.fit(texts, labels)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)

    print("[STARTUP] ✅ Model trained and saved!")
    return pipeline


def load_or_train():
    global tfidf_pipeline
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                tfidf_pipeline = pickle.load(f)
            print(f"[STARTUP] ✅ Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"[STARTUP] Load failed ({e}), retraining...")
            tfidf_pipeline = train_and_save()
    else:
        tfidf_pipeline = train_and_save()


# ── Run on import (Render calls this when the module loads) ──
load_or_train()


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "name"        : "Fake News Detection API",
        "status"      : "running",
        "model_loaded": tfidf_pipeline is not None,
        "endpoints"   : {
            "POST /predict": "Send { text: '...' } to get credibility score",
            "GET  /health" : "Health check"
        }
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status"      : "ok",
        "model_loaded": tfidf_pipeline is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    if tfidf_pipeline is None:
        return jsonify({"error": "Model not ready yet, try again in 10 seconds"}), 503

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Send JSON with key 'text'"}), 400

    raw_text = data['text'].strip()
    if not raw_text:
        return jsonify({"error": "Text is empty"}), 400

    clean        = clean_text(raw_text)
    prob_fake    = float(tfidf_pipeline.predict_proba([clean])[0, 1])
    label        = "FAKE" if prob_fake >= 0.5 else "REAL"

    if prob_fake >= 0.75:
        credibility = "LOW"
        message     = "Strong signs of misinformation. Verify with trusted sources."
    elif prob_fake >= 0.5:
        credibility = "MEDIUM-LOW"
        message     = "May contain misleading content. Double-check before sharing."
    elif prob_fake >= 0.25:
        credibility = "MEDIUM-HIGH"
        message     = "Appears mostly credible. Check key claims independently."
    else:
        credibility = "HIGH"
        message     = "Appears credible based on language patterns."

    return jsonify({
        "label"      : label,
        "confidence" : round(prob_fake, 4),
        "credibility": credibility,
        "message"    : message,
    })


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    print(f"[API] Starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
