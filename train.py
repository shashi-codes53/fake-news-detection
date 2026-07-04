"""
Main Training Script — Fake News Detection
Trains both TF-IDF + LogReg (baseline) and BERT (advanced) models.

Usage:
    # Train only TF-IDF (fast, no GPU needed):
    python train.py --model tfidf --dataset kaggle

    # Train only BERT (slower, GPU recommended):
    python train.py --model bert --dataset kaggle

    # Train both:
    python train.py --model both --dataset kaggle
"""

import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset    import load_kaggle_fakenews, load_liar, get_texts_and_labels
from models.tfidf_model import TFIDFModel
from models.bert_model  import BERTFakeNewsModel
from utils.metrics      import compute_metrics, print_metrics, plot_results


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   choices=["tfidf", "bert", "both"], default="tfidf",
                        help="Which model to train")
    parser.add_argument("--dataset", choices=["kaggle", "liar"], default="kaggle",
                        help="Which dataset to use")
    parser.add_argument("--epochs",  type=int, default=4,
                        help="BERT training epochs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr",      type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=256,
                        help="Max token length for BERT")
    return parser.parse_args()


def load_data(dataset_name):
    """Load and return (train_df, val_df, test_df)."""
    print(f"\n[Data] Loading {dataset_name} dataset...")
    if dataset_name == "kaggle":
        return load_kaggle_fakenews()
    elif dataset_name == "liar":
        return load_liar()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_text_col(dataset_name):
    return 'text' if dataset_name == 'kaggle' else 'clean_statement'


def main():
    args = get_args()
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/logs",        exist_ok=True)

    # ── Load data ──
    train_df, val_df, test_df = load_data(args.dataset)
    text_col  = get_text_col(args.dataset)

    train_texts, train_labels = get_texts_and_labels(train_df, text_col)
    val_texts,   val_labels   = get_texts_and_labels(val_df,   text_col)
    test_texts,  test_labels  = get_texts_and_labels(test_df,  text_col)

    tfidf_preds = bert_preds = None
    tfidf_probs = bert_probs = None

    # ── TF-IDF + Logistic Regression ──
    if args.model in ["tfidf", "both"]:
        print("\n" + "="*55)
        print("  Training Model 1: TF-IDF + Logistic Regression")
        print("="*55)
        tfidf = TFIDFModel(max_features=50000, ngram_range=(1, 2))
        tfidf.train(train_texts, train_labels)
        tfidf.evaluate(val_texts, val_labels, "Validation")
        tfidf_results = tfidf.evaluate(test_texts, test_labels, "Test")
        tfidf.get_top_features(n=15)
        tfidf.save("outputs/checkpoints/tfidf_model.pkl")

        tfidf_preds = tfidf.predict(test_texts)
        tfidf_probs = tfidf.predict_proba(test_texts)

        metrics = compute_metrics(test_labels, tfidf_preds, tfidf_probs)
        print_metrics(metrics, "TF-IDF + LogReg")

    # ── BERT ──
    if args.model in ["bert", "both"]:
        print("\n" + "="*55)
        print("  Training Model 2: BERT (Fine-tuned)")
        print("="*55)
        bert = BERTFakeNewsModel(
            max_length=args.max_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr
        )
        bert.train(train_texts, train_labels, val_texts, val_labels)
        bert.evaluate(test_texts, test_labels, "Test")
        bert.save("outputs/checkpoints/bert_best.pt")

        bert_preds, bert_probs = bert.predict(test_texts)
        metrics = compute_metrics(test_labels, bert_preds, bert_probs)
        print_metrics(metrics, "BERT")

    # ── Comparison plot (if both trained) ──
    if args.model == "both" and tfidf_preds is not None and bert_preds is not None:
        print("\n[Plotting] Generating comparison plots...")
        plot_results(
            labels      = test_labels,
            tfidf_preds = tfidf_preds,
            bert_preds  = bert_preds,
            tfidf_probs = tfidf_probs,
            bert_probs  = bert_probs,
            save_path   = "outputs/comparison.png"
        )
        print("Done! Open outputs/comparison.png to see results.")

    print("\n✅ Training complete!")
    print("   Checkpoints saved in: outputs/checkpoints/")


if __name__ == "__main__":
    main()
