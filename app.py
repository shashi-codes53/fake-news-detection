"""
Flask API — Fake News Detection
Fixed for Render deployment.
"""

import os
import sys
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.dataset import clean_text

app = Flask(__name__)
CORS(app)

# ── Train model immediately at import time ──
print("[STARTUP] Initializing model...")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

TEXTS = [
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
    "nasa successfully launches new satellite to monitor earth climate patterns from orbit",
    "federal reserve adjusts interest rates in response to inflation and economic data",
    "new education policy aims to improve literacy rates in underprivileged communities",
    "world health organization updates guidelines on nutrition and physical activity globally",
    "SHOCKING secret they dont want you to know deep state exposed wake up now",
    "BREAKING government hiding truth about vaccines dangerous side effects covered up lies",
    "URGENT share before deleted illuminati controls media everything is fake news conspiracy",
    "celebrities caught in massive scandal pedophile ring cover up exposed truth finally revealed",
    "miracle cure doctors hate this one weird trick cures cancer instantly at home free",
    "you wont believe what they found alien bodies government hiding area 51 proof leaked",
    "FRAUD election stolen millions of votes evidence proof watch before banned and censored",
    "they are poisoning water supply chemtrails mind control exposed microchips in vaccines",
    "billionaire elite plan to control world population secret meeting leaked documents exposed",
    "mainstream media lying to you real truth hidden from public view wake up sheeple now",
    "BOMBSHELL whistleblower exposes government plot to microchip all citizens forced vaccines",
    "new world order plan revealed globalist agenda to enslave humanity exposed finally truth",
    "crisis actor caught on camera fake shooting staged by government for gun control agenda",
    "5g towers causing coronavirus spread truth they dont want you to know share before deleted",
    "george soros paying protesters to destroy america open borders globalist puppet master plan",
    "moon landing was faked nasa admits in secret documents leaked by insider whistleblower proof",
    "fluoride in water lowers iq government deliberately poisoning citizens since 1950s truth",
    "deep state planning false flag attack to start world war three imminent warning share now",
    "secret cure for all diseases suppressed by big pharma to keep patients paying money forever",
    "reptilian shapeshifters control world governments exposed in leaked footage share before ban",
]
LABELS = [0]*20 + [1]*20

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000, ngram_range=(1,2),
        stop_words='english', sublinear_tf=True, min_df=1,
    )),
    ('clf', LogisticRegression(max_iter=1000, C=1.0))
])
pipeline.fit(TEXTS, LABELS)
print("[STARTUP] ✅ Model ready!")


# ── Routes ──
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "name"        : "Fake News Detection API",
        "status"      : "running",
        "model_loaded": True,
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model_loaded": True})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Send JSON with key 'text'"}), 400

    raw_text = data.get('text', '').strip()
    if not raw_text:
        return jsonify({"error": "Text is empty"}), 400

    clean_t   = clean_text(raw_text)
    prob_fake = float(pipeline.predict_proba([clean_t])[0, 1])
    label     = "FAKE" if prob_fake >= 0.5 else "REAL"

    if prob_fake >= 0.75:
        credibility, message = "LOW",         "Strong signs of misinformation."
    elif prob_fake >= 0.5:
        credibility, message = "MEDIUM-LOW",  "May contain misleading content. Verify before sharing."
    elif prob_fake >= 0.25:
        credibility, message = "MEDIUM-HIGH", "Appears mostly credible. Check key claims."
    else:
        credibility, message = "HIGH",        "Appears credible based on language patterns."

    return jsonify({
        "label"      : label,
        "confidence" : round(prob_fake, 4),
        "credibility": credibility,
        "message"    : message,
    })


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
