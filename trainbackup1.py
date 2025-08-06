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
BATCH_SIZE = 4
NUM_EPOCHS = 10
LR = 1e-4
NUM_WORKERS = 2  # Safe for Windows; increase on Linux if possible
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "best_model.pt"

# === Compute Positive Weights ===
def compute_pos_weights(dataset):
    print("[INFO] Computing pos_weights (no multiprocessing)...")
    all_labels = []
    loader = DataLoader(dataset, batch_size=8, num_workers=0)  # Avoid spawn crash on Windows
    for _, y in loader:
        all_labels.append(y)
    labels = torch.cat(all_labels, dim=0)
    pos_counts = labels.sum(dim=0)
    neg_counts = labels.shape[0] - pos_counts
    return neg_counts / (pos_counts + 1e-5)

# === Evaluation ===
def evaluate(model, loader):
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)

    aucs = []
    for i in range(14):
        y_col, y_pred_col = y_true[:, i], y_pred[:, i]
        if len(np.unique(y_col)) < 2:
            auc = float('nan')
        else:
            auc = roc_auc_score(y_col, y_pred_col)
        aucs.append(auc)

    auc_ap = aucs[0]
    location_aucs = aucs[1:14]
    location_aucs_valid = [a for a in location_aucs if not np.isnan(a)]

    if not np.isnan(auc_ap) and location_aucs_valid:
        mean_location_auc = sum(location_aucs_valid) / 13
        final_score = 0.5 * (auc_ap + mean_location_auc)
    else:
        final_score = float('nan')

    return final_score, aucs


# === Main Training Loop ===
if __name__ == "__main__":
    print("[INFO] Preparing data...")
    all_uids = get_series_uids_with_labels(LABELS_CSV)
    np.random.seed(42)
    np.random.shuffle(all_uids)
    split = int(0.8 * len(all_uids))
    train_uids, val_uids = all_uids[:split], all_uids[split:]

    train_dataset = RSNADataset(SERIES_DIR, LABELS_CSV, train_uids, input_size=INPUT_SIZE)
    val_dataset = RSNADataset(SERIES_DIR, LABELS_CSV, val_uids, input_size=INPUT_SIZE)

    pos_weight = compute_pos_weights(train_dataset).to(DEVICE)
    print(f"[INFO] Pos Weight:\n{pos_weight}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print("[INFO] Initializing model...")
    model = ResNet3D(num_outputs=14).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = 0
    count = 0
    for epoch in range(NUM_EPOCHS):
        count +=1
        print(f"\n[Epoch {epoch+1}/{NUM_EPOCHS}] Training...")
        model.train()
        running_loss = 0.0

        for x, y in tqdm(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_auc, per_class_aucs = evaluate(model, val_loader)
        
        log_msg = f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f} | Val Weighted AUC: {val_auc:.4f}"
        print(f"\n{log_msg}")
        with open("training_log.txt", "a") as f:
            f.write(log_msg + "\n")

        # Save checkpoint for every epoch
        epoch_save_path = f"model_epoch_{count}.pt"
        torch.save(model.state_dict(), epoch_save_path)
        print(f"📦 Model saved: {epoch_save_path}")

        # Also track the best one
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), SAVE_PATH)
            print("✅ New best model saved.")


    print(f"\n✅ Training complete. Best Validation Weighted AUC: {best_auc:.4f}")
