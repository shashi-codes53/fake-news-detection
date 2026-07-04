"""
Evaluation Metrics for Fake News Detection
Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report
)


def compute_metrics(labels, preds, probs=None):
    """
    Compute all classification metrics.

    Args:
        labels : ground truth (list of 0/1)
        preds  : predicted labels (list of 0/1)
        probs  : predicted probabilities for class 1 (optional, for AUC)

    Returns:
        dict of metric names → values
    """
    metrics = {
        "accuracy" : accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall"   : recall_score(labels, preds, zero_division=0),
        "f1"       : f1_score(labels, preds, zero_division=0),
    }
    if probs is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(labels, probs)
        except Exception:
            metrics["roc_auc"] = 0.0

    return metrics


def print_metrics(metrics, model_name="Model"):
    """Pretty-print metrics table."""
    print(f"\n{'='*45}")
    print(f"  Results: {model_name}")
    print(f"{'='*45}")
    for name, val in metrics.items():
        print(f"  {name.upper():12s}: {val:.4f}")
    print(f"{'='*45}\n")


def plot_results(labels, tfidf_preds, bert_preds,
                 tfidf_probs=None, bert_probs=None,
                 save_path="outputs/evaluation.png"):
    """
    Plots:
    1. Confusion matrices for both models side by side
    2. ROC curves for both models (if probs provided)
    3. Metric comparison bar chart
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Fake News Detection — Model Evaluation", fontsize=16, fontweight='bold')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── Confusion matrices ──
    for idx, (model_preds, model_name) in enumerate(
        [(tfidf_preds, "TF-IDF + LogReg"), (bert_preds, "BERT")]
    ):
        ax = fig.add_subplot(gs[0, idx])
        cm = confusion_matrix(labels, model_preds)
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(f"Confusion Matrix\n{model_name}", fontsize=12)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Real', 'Fake']); ax.set_yticklabels(['Real', 'Fake'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=14, color='white' if cm[i, j] > cm.max()/2 else 'black')
        plt.colorbar(im, ax=ax)

    # ── ROC Curves ──
    if tfidf_probs is not None and bert_probs is not None:
        ax = fig.add_subplot(gs[0, 2])
        for probs, name, color in [
            (tfidf_probs, "TF-IDF + LogReg", "steelblue"),
            (bert_probs,  "BERT",            "darkorange")
        ]:
            fpr, tpr, _ = roc_curve(labels, probs)
            auc = roc_auc_score(labels, probs)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
        ax.plot([0,1],[0,1], 'k--', lw=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves"); ax.legend(fontsize=9)

    # ── Metric Comparison Bar Chart ──
    ax = fig.add_subplot(gs[1, :])
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    tfidf_vals = [
        accuracy_score(labels, tfidf_preds),
        precision_score(labels, tfidf_preds, zero_division=0),
        recall_score(labels, tfidf_preds, zero_division=0),
        f1_score(labels, tfidf_preds, zero_division=0),
    ]
    bert_vals = [
        accuracy_score(labels, bert_preds),
        precision_score(labels, bert_preds, zero_division=0),
        recall_score(labels, bert_preds, zero_division=0),
        f1_score(labels, bert_preds, zero_division=0),
    ]
    x = np.arange(len(metric_names))
    w = 0.35
    bars1 = ax.bar(x - w/2, tfidf_vals, w, label='TF-IDF + LogReg', color='steelblue', alpha=0.85)
    bars2 = ax.bar(x + w/2, bert_vals,  w, label='BERT',            color='darkorange', alpha=0.85)
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, 1.1); ax.set_xticks(x); ax.set_xticklabels(metric_names, fontsize=12)
    ax.set_ylabel("Score"); ax.set_title("Model Comparison — All Metrics")
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Saved] Evaluation plot → {save_path}")
    plt.close()
