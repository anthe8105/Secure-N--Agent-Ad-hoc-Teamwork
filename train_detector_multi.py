"""
train_detector_multi.py
-----------------------
Trains DBBCMultiObserver (full DBBC with N>1 strategic agents + PoE fusion)
on multi-observer (state, action) sequence data from collect_multi_data.py.

Loss: BCE(shared_belief, y) + λ·MSE(shared_belief, y)   (paper Eq. 8)

Usage:
  python train_detector_multi.py --csv results/multi_state_data_14_<exp>.csv
  python train_detector_multi.py --csv <csv> --train-ratio 0.7 --epochs 100
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import glob
import os
import argparse
from sklearn.metrics import accuracy_score, f1_score

from src.models.dbbc import DBBCMultiObserver
from src.data.dataset import get_multi_train_test_dataloaders

# ---------- Hyperparameters ----------
HIDDEN_DIM   = 64
SEQ_LEN      = 10
BATCH_SIZE   = 32
EPOCHS       = 100
LAMBDA_CAL   = 0.1
LR           = 1e-3
SAVE_PATH    = "multi_detector_pretrained.pth"
EVAL_EVERY   = 10
REL_FEAT_DIM = 3


def evaluate_loader(model, loader, lambda_cal=LAMBDA_CAL):
    """Run inference on a DataLoader; return (loss, accuracy, f1)."""
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            seqs = batch['seqs']              # (B, N, H, D)
            rel  = batch['rel_features']      # (B, N, R)
            labels = batch['label']           # (B, 1)

            shared_belief, _, _ = model(seqs, rel)

            l_det = F.binary_cross_entropy(shared_belief, labels)
            l_cal = F.mse_loss(shared_belief, labels)
            total_loss += (l_det + lambda_cal * l_cal).item()
            n_batches += 1

            preds = (shared_belief > 0.5).float()
            all_preds.extend(preds.numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    avg_loss = total_loss / n_batches if n_batches > 0 else float('nan')
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, acc, f1


def train(override_csv=None, train_ratio=0.7, epochs=EPOCHS):
    # Resolve CSV
    if override_csv is not None:
        csv_path = override_csv
    else:
        csv_files = glob.glob("results/multi_state_data_*.csv")
        if not csv_files:
            print("No multi_state_data CSV found. Run collect_multi_data.py first.")
            return
        csv_path = max(csv_files, key=os.path.getctime)

    print(f"Training on   : {csv_path}")
    print(f"Split         : {int(train_ratio*100)}% train / "
          f"{int((1-train_ratio)*100)}% test  (temporal, gap = {SEQ_LEN} steps)")

    # Build split dataloaders
    train_loader, test_loader = get_multi_train_test_dataloaders(
        csv_path, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, train_ratio=train_ratio)

    if len(train_loader) == 0:
        print("ERROR: training set is empty.")
        return

    # Infer dimensions from first batch
    sample_batch = next(iter(train_loader))
    n_observers = sample_batch['seqs'].shape[1]
    input_dim   = sample_batch['seqs'].shape[3]
    print(f"\nN observers   : {n_observers}")
    print(f"Input dim     : {input_dim}  (obs_dim + action_dim)")
    print(f"Rel feat dim  : {REL_FEAT_DIM}")

    # Model + optimizer
    model = DBBCMultiObserver(
        n_observers=n_observers,
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        rel_feat_dim=REL_FEAT_DIM
    )
    optimizer = optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params  : {total_params:,}")

    print(f"\n{'Epoch':>6}  {'Train Loss':>12}  {'Test Loss':>10}  "
          f"{'Test Acc':>10}  {'Test F1':>8}")
    print("-" * 60)

    best_f1 = -1.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0

        for batch in train_loader:
            seqs   = batch['seqs']
            rel    = batch['rel_features']
            labels = batch['label']

            optimizer.zero_grad()
            shared_belief, _, _ = model(seqs, rel)

            l_det = F.binary_cross_entropy(shared_belief, labels)
            l_cal = F.mse_loss(shared_belief, labels)
            loss  = l_det + LAMBDA_CAL * l_cal
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        train_loss = epoch_loss / batches if batches > 0 else float('nan')

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == epochs - 1:
            test_loss, test_acc, test_f1 = evaluate_loader(model, test_loader)
            print(f"{epoch+1:>6}  {train_loss:>12.4f}  {test_loss:>10.4f}  "
                  f"{test_acc:>10.4f}  {test_f1:>8.4f}")
            if test_f1 > best_f1:
                best_f1 = test_f1
                torch.save(model.state_dict(), SAVE_PATH)
        else:
            print(f"{epoch+1:>6}  {train_loss:>12.4f}")

    print(f"\nBest test F1  : {best_f1:.4f}")
    print(f"Saved model   → {SAVE_PATH}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train DBBCMultiObserver with temporal train/test split")
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to multi_state_data CSV')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Fraction of steps for training (default 0.7)')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of training epochs (default {EPOCHS})')
    args = parser.parse_args()
    train(override_csv=args.csv, train_ratio=args.train_ratio, epochs=args.epochs)
