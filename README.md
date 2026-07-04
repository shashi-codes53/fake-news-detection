# Fake News Detection System

An ensemble classifier combining TF-IDF + Logistic Regression and a fine-tuned BERT model on the LIAR dataset. Deployed as a Chrome extension (Flask backend) that flags news article credibility in real time.

## Project Structure

```
fake_news_detection/
├── models/
│   ├── tfidf_model.py     # TF-IDF + Logistic Regression
│   └── bert_model.py      # Fine-tuned BERT classifier
├── data/
│   └── dataset.py         # LIAR + Kaggle dataset loaders
├── utils/
│   └── metrics.py         # Accuracy, F1, ROC-AUC, plots
├── extension/
│   ├── manifest.json      # Chrome extension config
│   └── popup.html         # Extension UI + JS
├── train.py               # Train TF-IDF and/or BERT
├── app.py                 # Flask API backend
├── requirements.txt
└── outputs/
    ├── checkpoints/       # Saved models
    └── comparison.png     # Evaluation plots
```

## How It Works

**Model 1 — TF-IDF + Logistic Regression (Baseline)**
Converts articles into word-frequency vectors, then trains Logistic Regression to classify fake vs real. Fast, interpretable, no GPU needed. Achieves ~85-88% accuracy.

**Model 2 — BERT (Fine-tuned)**
Fine-tunes `bert-base-uncased` on the fake news dataset. BERT reads the full context of sentences (not just word counts), achieving ~89% accuracy. Requires GPU for fast training.

**Chrome Extension + Flask API**
The Flask API loads the trained model and exposes a `/predict` endpoint. The Chrome extension grabs the page text and calls the API, then shows a real-time credibility badge.

## Setup — Step by Step

### 1. Clone and install
```bash
git clone <your-repo>
cd fake_news_detection
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 2. Get the dataset

**Option A — Kaggle (easier, bigger dataset)**
- Go to: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- Download → unzip → put `True.csv` and `Fake.csv` inside `data/`

**Option B — LIAR dataset**
- Go to: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
- Download → unzip → put `train.tsv`, `valid.tsv`, `test.tsv` inside `data/liar/`

### 3. Train TF-IDF model (fast, ~2 minutes, no GPU)
```bash
python train.py --model tfidf --dataset kaggle
```

### 4. Train BERT model (slow, GPU recommended)
```bash
python train.py --model bert --dataset kaggle --epochs 4
```

### 5. Train both and compare
```bash
python train.py --model both --dataset kaggle
```
Opens `outputs/comparison.png` with side-by-side metrics.

### 6. Start the Flask API
```bash
python app.py
```
API runs on `http://localhost:5000`

Test it manually:
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"Scientists discover breakthrough cancer treatment\"}"
```

### 7. Load the Chrome Extension
1. Open Chrome → go to `chrome://extensions/`
2. Toggle **Developer Mode** ON (top right)
3. Click **Load unpacked**
4. Select the `extension/` folder
5. The extension icon appears in your toolbar
6. Make sure `app.py` is running, then click the extension on any news page

## Results

| Model              | Accuracy | F1    |
|--------------------|----------|-------|
| TF-IDF + LogReg    | ~87%     | ~0.87 |
| BERT (fine-tuned)  | ~89%     | ~0.89 |
