"""
train_detector.py
-----------------
Trains DBBCDetector on raw (state, action) sequences with TEMPORAL train/test split.

Why temporal split (not random)?
  Adjacent windows share H-1 steps. A random split would put overlapping
  windows in both train and test, making the "test" score meaningless
  (the model has literally seen most of that window during training).

Split strategy (per agent):
  steps  0 .. floor(n * train_ratio) - 1           → train windows
  steps  floor(n * train_ratio) .. cut + H - 1     → GAP (no windows)
  steps  floor(n * train_ratio) + H .. n - 1       → test  windows

The H-step gap guarantees zero step-level overlap between any train window
and any test window.

Usage:
  python train_detector.py --csv results/state_data_13_<exp>.csv
  python train_detector.py --csv results/state_data_13_<exp>.csv --train-ratio 0.7
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import glob
import os
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.models.dbbc import DBBCDetector
from src.data.dataset import get_train_test_dataloaders

OBS_DIM      = 10
ACTION_DIM   = 5
INPUT_DIM    = OBS_DIM + ACTION_DIM  # 15
HIDDEN_DIM   = 32
SEQ_LEN      = 10
BATCH_SIZE   = 32
EPOCHS       = 100
LAMBDA_CAL   = 0.1
LR           = 1e-3
SAVE_PATH    = "detector_pretrained.pth"
EVAL_EVERY   = 10   # print test metrics every N epochs


def evaluate_loader(model, loader):
    """Run inference on a DataLoader; return (loss, accuracy, f1)."""
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    n_batches = 0

    with torch.no_grad():
        for seq, labels in loader:
            belief = model(seq)
            l_det = F.binary_cross_entropy(belief, labels)
            l_cal = F.mse_loss(belief, labels)
            total_loss += (l_det + LAMBDA_CAL * l_cal).item()
            n_batches += 1

            preds = (belief > 0.5).float()
            all_preds.extend(preds.numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    avg_loss = total_loss / n_batches if n_batches > 0 else float('nan')
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, acc, f1


def train(override_csv=None, train_ratio=0.7):
    # Resolve CSV
    if override_csv is not None:
        csv_path = override_csv
    else:
        csv_files = glob.glob("results/state_data_*.csv")
        if not csv_files:
            print("No state_data CSV found. Run collect_state_data.py first.")
            return
        csv_path = max(csv_files, key=os.path.getctime)

    print(f"Training on  : {csv_path}")
    print(f"Split        : {int(train_ratio*100)}% train / "
          f"{int((1-train_ratio)*100)}% test  (temporal, gap = {SEQ_LEN} steps)")

    # Build split dataloaders
    train_loader, test_loader = get_train_test_dataloaders(
        csv_path, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, train_ratio=train_ratio)

    if len(train_loader) == 0:
        print("ERROR: training set is empty. Collect more data or reduce train_ratio.")
        return
    if len(test_loader) == 0:
        print("WARNING: test set is empty. Collect more data or increase train_ratio.")

    # Model + optimizer
    model = DBBCDetector(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\nDBBCDetector: input_dim={INPUT_DIM}, hidden_dim={HIDDEN_DIM}, seq_len={SEQ_LEN}")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Test Loss':>10}  "
          f"{'Test Acc':>10}  {'Test F1':>8}")
    print("-" * 60)

    best_f1 = -1.0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        batches = 0

        for seq, labels in train_loader:
            optimizer.zero_grad()
            belief = model(seq)
            l_det = F.binary_cross_entropy(belief, labels)
            l_cal = F.mse_loss(belief, labels)
            loss  = l_det + LAMBDA_CAL * l_cal
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        train_loss = epoch_loss / batches if batches > 0 else float('nan')

        # Evaluate on test set every EVAL_EVERY epochs
        if (epoch + 1) % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
            test_loss, test_acc, test_f1 = evaluate_loader(model, test_loader)
            print(f"{epoch+1:>6}  {train_loss:>12.4f}  {test_loss:>10.4f}  "
                  f"{test_acc:>10.4f}  {test_f1:>8.4f}")

            # Save the best checkpoint by test F1
            if test_f1 > best_f1:
                best_f1 = test_f1
                torch.save(model.state_dict(), SAVE_PATH)
        else:
            print(f"{epoch+1:>6}  {train_loss:>12.4f}")

    print(f"\nBest test F1 : {best_f1:.4f}")
    print(f"Saved model  → {SAVE_PATH}  (best checkpoint by test F1)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train DBBCDetector with temporal train/test split")
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to state_data CSV')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Fraction of steps used for training (default 0.7)')
    args = parser.parse_args()
    train(override_csv=args.csv, train_ratio=args.train_ratio)
