import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from config import (
    DEVICE,
    NUM_CLASSES,
    LSTM_LR,
    LSTM_PATIENCE,
    PLOT_DIR
)

matplotlib.use("Agg")

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
# TRAIN 1 EPOCH
# =========================
def train_one_epoch(model, dataloader, optimizer, criterion):
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
# EVALUATE
# =========================
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, labels, lengths in dataloader:
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
# TRAIN MODEL 
# =========================
def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    save_path,
    epochs,
    lr,
    patience,
    model_name="model"   # 🔥 TAMBAH INI
):

    os.makedirs(PLOT_DIR, exist_ok=True)

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    print(f"\n🚀 Training {model_name}...\n")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⛔ Early stopping!")
                break

        # =========================
    # TEST
    # =========================
    print(f"\n📊 Evaluasi {model_name}...")

    _, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion)

    print("\n📄 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=LABEL_LIST))

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred, average="weighted"))

    # =========================
    # CONFUSION MATRIX
    # =========================
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=LABEL_LIST,
                yticklabels=LABEL_LIST)

    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_path = os.path.join(PLOT_DIR, f"confusion_matrix_{model_name}.png")
    plt.savefig(cm_path)
    plt.close()

    print(f"📊 Saved: {cm_path}")

    # =========================
    # TRAINING CURVES
    # =========================
    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs_range, history["train_loss"], label="Train")
    plt.plot(epochs_range, history["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs_range, history["train_acc"], label="Train")
    plt.plot(epochs_range, history["val_acc"], label="Val")
    plt.title("Accuracy")
    plt.legend()

    plt.suptitle(f"Training Curves — {model_name}")

    curve_path = os.path.join(PLOT_DIR, f"training_curves_{model_name}.png")
    plt.savefig(curve_path)
    plt.close()

    print(f"📊 Saved: {curve_path}")

    # =========================
    # METRICS UNTUK COMPARISON
    # =========================
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    metrics = {
        "accuracy": test_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }

    return metrics