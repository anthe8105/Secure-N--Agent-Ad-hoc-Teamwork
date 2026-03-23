"""
evaluate_detector.py
--------------------
Evaluates the pretrained DBBCDetector model on raw (state, action) sequence data.

Usage:
  python evaluate_detector.py --csv results/state_data_13_<exp>.csv
"""
import torch
import glob
import os
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.models.dbbc import DBBCDetector
from src.data.dataset import get_state_action_dataloader

OBS_DIM = 10
ACTION_DIM = 5
INPUT_DIM = OBS_DIM + ACTION_DIM  # 15
HIDDEN_DIM = 32
SEQ_LEN = 10
BATCH_SIZE = 32
LOAD_PATH = "detector_pretrained.pth"


def evaluate(override_csv=None):
    # Resolve CSV
    if override_csv is not None:
        csv_path = override_csv
    else:
        csv_files = glob.glob("results/state_data_*.csv")
        if not csv_files:
            print("No state_data CSV found. Run collect_state_data.py first.")
            return
        csv_path = max(csv_files, key=os.path.getctime)

    print(f"Evaluating on: {csv_path}")

    if not os.path.exists(LOAD_PATH):
        print(f"{LOAD_PATH} not found. Please train the model first.")
        return

    model = DBBCDetector(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
    model.load_state_dict(torch.load(LOAD_PATH, map_location='cpu'))
    model.eval()

    dataloader = get_state_action_dataloader(csv_path, batch_size=BATCH_SIZE, seq_len=SEQ_LEN)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for seq, labels in dataloader:
            belief = model(seq)               # (batch, 1)
            preds = (belief > 0.5).float()
            all_preds.extend(preds.numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    if not all_preds:
        print("No evaluation data found.")
        return

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    adv_windows = sum(all_labels)
    total = len(all_labels)

    print("\n--- DBBCDetector Evaluation Results ---")
    print(f"Total windows evaluated : {total}")
    print(f"Adversary windows       : {int(adv_windows)} ({100*adv_windows/total:.1f}%)")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate DBBCDetector")
    parser.add_argument('--csv', type=str, default=None, help='Path to state_data CSV')
    args = parser.parse_args()
    evaluate(override_csv=args.csv)
