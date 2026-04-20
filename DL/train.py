import os
import time
import copy

import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

from config import (
    DEVICE, NUM_CLASSES,
    LSTM_LR, LSTM_PATIENCE, BERT_LR, BERT_PATIENCE,
    PLOT_DIR,
)

matplotlib.use("Agg")

# =========================
# LABEL LIST (IMDB)
# =========================
LABEL_LIST = ["negative", "positive"]

# =========================
# SEED
# =========================
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# LOSS
# =========================
def get_criterion():
    return nn.CrossEntropyLoss()


# =========================
# TRAIN LSTM
# =========================
def train_one_epoch_lstm(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, labels, lengths in dataloader:
        x, labels, lengths = x.to(DEVICE), labels.to(DEVICE), lengths.to(DEVICE)

        optimizer.zero_grad()

        if hasattr(model, "attention"):
            logits, _ = model(x, lengths)
        else:
            logits = model(x, lengths)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# =========================
# TRAIN BERT
# =========================
def train_one_epoch_bert(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# =========================
# EVALUATE
# =========================
def evaluate(model, dataloader, criterion, is_bert=False):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:

            if is_bert:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                logits = model(input_ids, attention_mask)

            else:
                x, labels, lengths = batch
                x, labels, lengths = x.to(DEVICE), labels.to(DEVICE), lengths.to(DEVICE)

                if hasattr(model, "attention"):
                    logits, _ = model(x, lengths)
                else:
                    logits = model(x, lengths)

            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# =========================
# TRAIN LOOP
# =========================
def train_model(
    model,
    train_loader,
    val_loader,
    model_type,
    save_path,
    epochs,
    lr,
    patience,
):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = get_criterion()

    is_bert = model_type == "bert"

    best_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        train_loss, train_acc = (
            train_one_epoch_bert(model, train_loader, optimizer, criterion)
            if is_bert else
            train_one_epoch_lstm(model, train_loader, optimizer, criterion)
        )

        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, is_bert)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss  : {val_loss:.4f} | Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping!")
                break

    return history


# =========================
# REPORT
# =========================
def print_report(y_true, y_pred):
    print(classification_report(y_true, y_pred, target_names=LABEL_LIST))

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    print("Accuracy:", acc)
    print("F1 Score:", f1)


# =========================
# CONFUSION MATRIX
# =========================
def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=LABEL_LIST,
                yticklabels=LABEL_LIST)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()