"""
Fine-tuned BERT Model for Fake News Detection
Uses Hugging Face Transformers — bert-base-uncased pretrained weights.

How BERT works:
- BERT is a large neural network pretrained on Wikipedia + BookCorpus.
- It already understands language context deeply.
- We add a single classification layer on top and fine-tune
  the whole model on our fake news dataset.
- This is called "transfer learning" — we reuse knowledge
  from a massive model instead of training from scratch.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


# ─────────────────────────────────────────────
# 1.  Dataset for BERT
# ─────────────────────────────────────────────
class FakeNewsDataset(Dataset):
    """
    Converts raw text into BERT input format:
    - input_ids     : token indices
    - attention_mask: 1 for real tokens, 0 for padding
    """
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids'     : encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label'         : torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ─────────────────────────────────────────────
# 2.  BERT Classifier
# ─────────────────────────────────────────────
class BERTClassifier(nn.Module):
    """
    BERT + Classification head.

    Architecture:
        Input text
          └─ BERT encoder (12 transformer layers)
              └─ [CLS] token representation  (768-dim vector)
                  └─ Dropout (0.3)
                      └─ Linear(768 → 2)    (Real or Fake)
                          └─ Softmax
    """
    def __init__(self, model_name='bert-base-uncased',
                 num_classes=2, dropout=0.3):
        super(BERTClassifier, self).__init__()
        self.bert    = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            self.bert.config.hidden_size, num_classes  # 768 → 2
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # [CLS] token is the first token — represents the whole sentence
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits     = self.classifier(cls_output)
        return logits


# ─────────────────────────────────────────────
# 3.  BERT Model Wrapper
# ─────────────────────────────────────────────
class BERTFakeNewsModel:
    """
    High-level wrapper for training, evaluating and predicting with BERT.
    """
    def __init__(self, model_name='bert-base-uncased',
                 max_length=256, batch_size=16, epochs=4, lr=2e-5):
        self.model_name  = model_name
        self.max_length  = max_length
        self.batch_size  = batch_size
        self.epochs      = epochs
        self.lr          = lr
        self.device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"[BERT Model] Using device: {self.device}")
        print(f"[BERT Model] Loading tokenizer: {model_name}")

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model     = BERTClassifier(model_name).to(self.device)

    def _make_loader(self, texts, labels, shuffle=True):
        dataset = FakeNewsDataset(texts, labels, self.tokenizer, self.max_length)
        return DataLoader(dataset, batch_size=self.batch_size,
                          shuffle=shuffle, num_workers=0)

    def train(self, train_texts, train_labels, val_texts, val_labels):
        train_loader = self._make_loader(train_texts, train_labels, shuffle=True)
        val_loader   = self._make_loader(val_texts,   val_labels,   shuffle=False)

        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0

        for epoch in range(1, self.epochs + 1):
            # ── Train ──
            self.model.train()
            total_loss, all_preds, all_labels = 0, [], []

            for batch_idx, batch in enumerate(train_loader):
                input_ids      = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels         = batch['label'].to(self.device)

                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss   = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

                if (batch_idx + 1) % 20 == 0:
                    print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} "
                          f"| Loss: {loss.item():.4f}")

            train_acc = accuracy_score(all_labels, all_preds)

            # ── Validate ──
            val_acc, val_loss = self._evaluate_loader(val_loader, criterion)

            print(f"\n{'─'*55}")
            print(f"  Epoch {epoch}/{self.epochs}")
            print(f"  Train → Loss: {total_loss/len(train_loader):.4f} | Acc: {train_acc:.4f}")
            print(f"  Val   → Loss: {val_loss:.4f}                | Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save("outputs/checkpoints/bert_best.pt")
                print(f"  ✅ New best val accuracy: {best_val_acc:.4f} — saved!")
            print(f"{'─'*55}\n")

        print(f"\n✅ BERT Training complete! Best Val Accuracy: {best_val_acc:.4f}")

    @torch.no_grad()
    def _evaluate_loader(self, loader, criterion):
        self.model.eval()
        total_loss, all_preds, all_labels = 0, [], []
        for batch in loader:
            input_ids      = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels         = batch['label'].to(self.device)
            logits = self.model(input_ids, attention_mask)
            loss   = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        return acc, total_loss / len(loader)

    @torch.no_grad()
    def predict(self, texts):
        """Predict labels for a list of texts."""
        self.model.eval()
        dummy_labels = [0] * len(texts)
        loader = self._make_loader(texts, dummy_labels, shuffle=False)
        all_preds, all_probs = [], []
        for batch in loader:
            input_ids      = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            logits = self.model(input_ids, attention_mask)
            probs  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # prob of fake
        return np.array(all_preds), np.array(all_probs)

    def evaluate(self, texts, labels, split_name="Test"):
        preds, probs = self.predict(texts)
        acc    = accuracy_score(labels, preds)
        report = classification_report(labels, preds, target_names=["Real", "Fake"])
        print(f"\n{'='*50}")
        print(f"  {split_name} Results — BERT")
        print(f"{'='*50}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"\n{report}")
        return {"accuracy": acc, "report": report}

    def save(self, path="outputs/checkpoints/bert_best.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'model_name' : self.model_name,
            'max_length' : self.max_length,
        }, path)

    def load(self, path="outputs/checkpoints/bert_best.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.to(self.device)
        print(f"[BERT Model] Loaded ← {path}")
