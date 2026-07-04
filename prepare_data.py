"""
Run this ONCE after downloading the dataset.
Verifies files are in the right place and shows a data summary.

Usage:
    # For Kaggle dataset (True.csv + Fake.csv):
    python prepare_data.py --dataset kaggle

    # For LIAR dataset (train.tsv, valid.tsv, test.tsv):
    python prepare_data.py --dataset liar
"""

import os
import sys
import argparse
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.dataset import load_kaggle_fakenews, load_liar, clean_text


def check_kaggle():
    print("\n[Kaggle] Checking files...")
    missing = []
    for f in ["data/True.csv", "data/Fake.csv"]:
        if os.path.exists(f):
            size = os.path.getsize(f) / (1024*1024)
            print(f"  ✅ {f}  ({size:.1f} MB)")
        else:
            print(f"  ❌ {f}  NOT FOUND")
            missing.append(f)

    if missing:
        print("\n  Download from:")
        print("  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        print("  Then put True.csv and Fake.csv inside the data/ folder.")
        return False

    print("\n[Kaggle] Loading and verifying data...")
    train_df, val_df, test_df = load_kaggle_fakenews()

    print("\n[Kaggle] Sample fake article:")
    fake_sample = train_df[train_df['label']==1].iloc[0]['text']
    print(f"  {fake_sample[:200]}...")

    print("\n[Kaggle] Sample real article:")
    real_sample = train_df[train_df['label']==0].iloc[0]['text']
    print(f"  {real_sample[:200]}...")

    print(f"\n✅ Dataset ready! Total: {len(train_df)+len(val_df)+len(test_df)} samples")
    return True


def check_liar():
    print("\n[LIAR] Checking files...")
    missing = []
    for f in ["data/liar/train.tsv", "data/liar/valid.tsv", "data/liar/test.tsv"]:
        if os.path.exists(f):
            size = os.path.getsize(f) / 1024
            print(f"  ✅ {f}  ({size:.0f} KB)")
        else:
            print(f"  ❌ {f}  NOT FOUND")
            missing.append(f)

    if missing:
        print("\n  Download from:")
        print("  https://www.cs.ucsb.edu/~william/data/liar_dataset.zip")
        print("  Then put train.tsv, valid.tsv, test.tsv inside data/liar/")
        return False

    print("\n[LIAR] Loading and verifying data...")
    train_df, val_df, test_df = load_liar()
    print(f"\n✅ LIAR dataset ready! Total: {len(train_df)+len(val_df)+len(test_df)} samples")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["kaggle", "liar"], default="kaggle")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    os.makedirs("data/liar", exist_ok=True)

    if args.dataset == "kaggle":
        ok = check_kaggle()
    else:
        ok = check_liar()

    if ok:
        print("\nNext step — train the model:")
        print("  python train.py --model tfidf --dataset", args.dataset)
