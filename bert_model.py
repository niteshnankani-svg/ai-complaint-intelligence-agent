import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os
import json

# ============================================================
# CONFIG
# ============================================================
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
RANDOM_STATE = 42
MIN_SAMPLES = 800  # minimum rows to train a category model

# Categories to merge into General (too few samples)
MERGE_INTO_GENERAL = ['Sports', 'Toys', 'Beauty']

# ============================================================
# STAGE 1 — Load and prepare data
# ============================================================
print("Loading dataset...")
df = pd.read_csv('flipkart_balanced.csv')

# Merge small categories into General
df['category'] = df['category'].apply(
    lambda x: 'General' if x in MERGE_INTO_GENERAL else x
)

# Convert sentiment to numbers — BERT needs numbers not strings
df['label'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

print(f"Total rows: {len(df):,}")
print(f"Categories: {df['category'].unique().tolist()}")

# ============================================================
# STAGE 2 — PyTorch Dataset class
# ============================================================
class ReviewDataset(Dataset):
    """
    Think of this as a smart list.
    PyTorch pulls items from it one by one during training.
    Each item = one tokenized review + its label.
    """
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize — converts text to BERT input format
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids':      encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label':          torch.tensor(label, dtype=torch.long)
        }
# ============================================================
# STAGE 3 — Training function
# ============================================================
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        input_ids =      batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels =         batch['label'].to(device)

        # Zero gradients from previous step
        optimizer.zero_grad()

        # Forward pass — BERT makes predictions
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        # Backward pass — BERT learns from mistakes
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total

# ============================================================
# STAGE 4 — Evaluation function
# ============================================================
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids =      batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels =         batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=['negative', 'positive']
    )
    return acc, report

# ============================================================
# MAIN — Train one model per category
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
os.makedirs('models', exist_ok=True)
results = {}

for category in df['category'].unique():
    print(f"\n{'='*50}")
    print(f"Training: {category}")
    print(f"{'='*50}")

    cat_df = df[df['category'] == category].reset_index(drop=True)

    if len(cat_df) < MIN_SAMPLES:
        print(f"  Skipping — only {len(cat_df)} rows (need {MIN_SAMPLES}+)")
        continue

    print(f"  Rows: {len(cat_df):,}")

    # Train/test split — 80% train, 20% test
    train_df, test_df = train_test_split(
        cat_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=cat_df['label']
    )

    # Create datasets
    train_dataset = ReviewDataset(
        train_df['input_text'].values,
        train_df['label'].values,
        tokenizer, MAX_LEN
    )
    test_dataset = ReviewDataset(
        test_df['input_text'].values,
        test_df['label'].values,
        tokenizer, MAX_LEN
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    # Load fresh BERT model for this category
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )
    model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    # Training loop
    best_acc = 0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        val_acc, val_report = evaluate(model, test_loader, device)

        print(f"  Epoch {epoch+1}/{EPOCHS} "
              f"| Loss: {train_loss:.4f} "
              f"| Train Acc: {train_acc:.4f} "
              f"| Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            model_path = f'models/{category}_bert'
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)

    # Final evaluation
    final_acc, final_report = evaluate(model, test_loader, device)
    results[category] = {
        'accuracy': round(final_acc, 4),
        'best_accuracy': round(best_acc, 4)
    }

    print(f"\n  Final Report for {category}:")
    print(final_report)

# Save results summary
with open('models/training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*50)
print("ALL MODELS TRAINED")
print("="*50)
for cat, res in results.items():
    print(f"  {cat}: Best Accuracy = {res['best_accuracy']}")
print("\nModels saved in ./models/ folder")