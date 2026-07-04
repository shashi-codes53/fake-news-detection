"""
Dataset loader for LIAR dataset + preprocessing utilities.

LIAR dataset has 6 labels:
  pants-fire, false, barely-true, half-true, mostly-true, true

We simplify to binary:
  FAKE = pants-fire, false, barely-true  → label 1
  REAL = half-true, mostly-true, true    → label 0
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# 1.  Text Cleaning
# ─────────────────────────────────────────────
def clean_text(text):
    """
    Clean raw news text:
    - Lowercase
    - Remove URLs
    - Remove special characters / extra spaces
    - Strip leading/trailing whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)          # remove URLs
    text = re.sub(r'[^a-z0-9\s]', ' ', text)            # keep letters & digits
    text = re.sub(r'\s+', ' ', text)                     # collapse spaces
    return text.strip()


# ─────────────────────────────────────────────
# 2.  LIAR Dataset Loader
# ─────────────────────────────────────────────
LIAR_COLUMNS = [
    'id', 'label', 'statement', 'subject', 'speaker',
    'speaker_job', 'state', 'party', 'barely_true_count',
    'false_count', 'half_true_count', 'mostly_true_count',
    'pants_fire_count', 'context'
]

# Map 6-class LIAR labels to binary
FAKE_LABELS = {'pants-fire', 'false', 'barely-true'}
REAL_LABELS = {'half-true', 'mostly-true', 'true'}

def load_liar(data_dir="data/liar"):
    """
    Load LIAR .tsv files and return binary-labelled DataFrames.

    Download from: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
    Expected files: train.tsv, valid.tsv, test.tsv
    """
    splits = {}
    for split in ['train', 'valid', 'test']:
        path = os.path.join(data_dir, f"{split}.tsv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"File not found: {path}\n"
                f"Download LIAR dataset from:\n"
                f"  https://www.cs.ucsb.edu/~william/data/liar_dataset.zip\n"
                f"Unzip into: {data_dir}/"
            )
        df = pd.read_csv(path, sep='\t', header=None, names=LIAR_COLUMNS)

        # Keep only binary-mappable labels
        df = df[df['label'].isin(FAKE_LABELS | REAL_LABELS)].copy()

        # Convert to binary
        df['binary_label'] = df['label'].apply(
            lambda x: 1 if x in FAKE_LABELS else 0
        )

        # Clean text
        df['clean_statement'] = df['statement'].apply(clean_text)

        splits[split] = df
        print(f"[LIAR] {split:6s}: {len(df)} samples  "
              f"(Fake: {df['binary_label'].sum()}, "
              f"Real: {(df['binary_label']==0).sum()})")

    return splits['train'], splits['valid'], splits['test']


# ─────────────────────────────────────────────
# 3.  Alternative: Load from CSV (Kaggle datasets)
# ─────────────────────────────────────────────
def load_kaggle_fakenews(true_csv="data/True.csv", fake_csv="data/Fake.csv"):
    """
    Load the popular Kaggle Fake News dataset.
    Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

    Files:
        data/True.csv  — real news articles
        data/Fake.csv  — fake news articles
    """
    for path in [true_csv, fake_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"File not found: {path}\n"
                f"Download from:\n"
                f"  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset\n"
                f"Put True.csv and Fake.csv inside data/"
            )

    true_df = pd.read_csv(true_csv)
    fake_df = pd.read_csv(fake_csv)

    true_df['label'] = 0  # real
    fake_df['label'] = 1  # fake

    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    # Combine title + text for richer features
    df['text'] = df.get('title', pd.Series(['']*len(df))).fillna('') + \
                 ' ' + df.get('text', pd.Series(['']*len(df))).fillna('')
    df['text'] = df['text'].apply(clean_text)

    # 80/10/10 split
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42,
                                         stratify=df['label'])
    val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=42,
                                         stratify=temp_df['label'])

    for name, d in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"[Kaggle] {name:6s}: {len(d)} samples  "
              f"(Fake: {d['label'].sum()}, Real: {(d['label']==0).sum()})")

    return train_df, val_df, test_df


# ─────────────────────────────────────────────
# 4.  Get arrays ready for models
# ─────────────────────────────────────────────
def get_texts_and_labels(df, text_col='text', label_col='label'):
    """Extract text list and label list from a DataFrame."""
    texts  = df[text_col].tolist()
    labels = df[label_col].tolist()
    return texts, labels


if __name__ == "__main__":
    # Test cleaning
    sample = "BREAKING: Government is LYING!!! Visit http://fakenews.com for truth!!!"
    print(f"Raw  : {sample}")
    print(f"Clean: {clean_text(sample)}")
