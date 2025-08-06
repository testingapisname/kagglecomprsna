import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
from dataset.rsna_dataset import RSNADataset, get_series_uids_with_labels
from models.resnet3d import ResNet3D
from tqdm import tqdm


# === CONFIG ===
SERIES_DIR = "data/rsna-intracranial-aneurysm-detection/series"
LABELS_CSV = "train.csv"
INPUT_SIZE = (64, 128, 128)
BATCH_SIZE = 2
NUM_EPOCHS = 3
LR = 1e-4
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "best_model.pt"

def compute_pos_weights(dataset):
    print("[INFO] Computing pos_weights using DataLoader (no multiprocessing)...")
    all_labels = []
    loader = DataLoader(dataset, batch_size=8, num_workers=0)
    for _, y in loader:
        all_labels.append(y)
    labels = torch.cat(all_labels, dim=0)
    pos_counts = labels.sum(dim=0)
    neg_counts = labels.shape[0] - pos_counts
    pos_weights = neg_counts / (pos_counts + 1e-5)
    return pos_weights


def evaluate(model, loader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)

    aucs = []
    for i in range(14):
        y_col = y_true[:, i]
        y_pred_col = y_pred[:, i]
        if len(np.unique(y_col)) < 2:
            auc = float('nan')
        else:
            auc = roc_auc_score(y_col, y_pred_col)
        aucs.append(auc)

    valid_location_aucs = [a for a in aucs[1:] if not np.isnan(a)]
    location_weight = len(valid_location_aucs)
    if np.isnan(aucs[0]):
        weighted_auc = np.nan
    else:
        weighted_auc = (13 * aucs[0] + sum(valid_location_aucs)) / (13 + location_weight)

    return weighted_auc, aucs


if __name__ == "__main__":
    print("[INFO] Loading UIDs...")
    all_uids = get_series_uids_with_labels(LABELS_CSV)
    np.random.seed(42)
    np.random.shuffle(all_uids)
    split = int(0.8 * len(all_uids))
    train_uids, val_uids = all_uids[:split], all_uids[split:]

    print("[INFO] Loading Datasets...")
    train_dataset = RSNADataset(SERIES_DIR, LABELS_CSV, train_uids, input_size=INPUT_SIZE)
    val_dataset = RSNADataset(SERIES_DIR, LABELS_CSV, val_uids, input_size=INPUT_SIZE)

    print("[INFO] Computing Class Weights...")
    pos_weight = compute_pos_weights(train_dataset).to(DEVICE)
    print(f"[INFO] Pos Weight:\n{pos_weight}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print("[INFO] Initializing Model...")
    model = ResNet3D(num_outputs=14).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        model.train()
        running_loss = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_auc, per_class_aucs = evaluate(model, val_loader)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Validation Weighted AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), SAVE_PATH)
            print("✅ Saved new best model!")
