"""
evaluate_detector_multi.py
--------------------------
Evaluates a trained DBBCMultiObserver on multi-observer data.

Reports overall and per-agent-type metrics (adversary vs teammate).

Usage:
  python evaluate_detector_multi.py --csv results/multi_state_data_14_<exp>.csv
"""
import torch
import argparse
import glob
import os
import pandas as pd
import ast
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

from src.models.dbbc import DBBCMultiObserver
from src.data.dataset import MultiObserverDataset, _multi_collate_fn
from torch.utils.data import DataLoader

MODEL_PATH   = "multi_detector_pretrained.pth"
HIDDEN_DIM   = 64
SEQ_LEN      = 10
REL_FEAT_DIM = 3


def evaluate(override_csv=None):
    # Resolve CSV
    if override_csv is not None:
        csv_path = override_csv
    else:
        csv_files = glob.glob("results/multi_state_data_*.csv")
        if not csv_files:
            print("No multi_state_data CSV found.")
            return
        csv_path = max(csv_files, key=os.path.getctime)

    print(f"Evaluating on: {csv_path}")

    # Load full dataset (no split — evaluate all windows)
    dataset = MultiObserverDataset(csv_path, seq_len=SEQ_LEN, split=None)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False,
                         drop_last=False, collate_fn=_multi_collate_fn)

    if len(dataset) == 0:
        print("ERROR: dataset is empty.")
        return

    # Infer model dimensions from data
    sample = dataset[0]
    n_observers = sample['seqs'].shape[0]
    input_dim   = sample['seqs'].shape[2]

    # Load model
    model = DBBCMultiObserver(
        n_observers=n_observers,
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        rel_feat_dim=REL_FEAT_DIM
    )

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: no checkpoint at {MODEL_PATH}. Train first.")
        return

    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    # Run inference
    all_preds, all_labels, all_beliefs = [], [], []
    with torch.no_grad():
        for batch in loader:
            seqs   = batch['seqs']
            rel    = batch['rel_features']
            labels = batch['label']

            shared_belief, fused_mu, fused_sigma = model(seqs, rel)

            preds = (shared_belief > 0.5).float()
            all_preds.extend(preds.numpy().flatten())
            all_labels.extend(labels.numpy().flatten())
            all_beliefs.extend(shared_belief.numpy().flatten())

    # Overall metrics
    print(f"\n--- DBBCMultiObserver Evaluation Results ---")
    print(f"Total windows evaluated : {len(all_labels)}")
    adv_count = sum(all_labels)
    print(f"Adversary windows       : {int(adv_count)} ({100*adv_count/len(all_labels):.1f}%)")
    print(f"Accuracy  : {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Precision : {precision_score(all_labels, all_preds, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(all_labels, all_preds, zero_division=0):.4f}")
    print(f"F1 Score  : {f1_score(all_labels, all_preds, zero_division=0):.4f}")

    # Belief distribution
    adv_beliefs = [b for b, y in zip(all_beliefs, all_labels) if y == 1.0]
    coop_beliefs = [b for b, y in zip(all_beliefs, all_labels) if y == 0.0]
    if adv_beliefs:
        print(f"\nAdversary belief  → mean={sum(adv_beliefs)/len(adv_beliefs):.4f}, "
              f"min={min(adv_beliefs):.4f}, max={max(adv_beliefs):.4f}")
    if coop_beliefs:
        print(f"Cooperative belief → mean={sum(coop_beliefs)/len(coop_beliefs):.4f}, "
              f"min={min(coop_beliefs):.4f}, max={max(coop_beliefs):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate DBBCMultiObserver")
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to multi_state_data CSV')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                        help=f'Path to saved model (default: {MODEL_PATH})')
    args = parser.parse_args()

    if args.model:
        MODEL_PATH = args.model

    evaluate(override_csv=args.csv)
