import torch
from src.models.dbbc import DBBCLocalEvidence, DBBCFusion
from src.data.dataset import get_dataloader
import numpy as np
import os
import glob
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(override_csv=None):
    # 1. Initialize DBBC Models
    obs_dim = 10
    action_dim = 5
    hidden_dim = 32
    rel_feat_dim = 8
    
    dbbc_local = DBBCLocalEvidence(obs_dim, action_dim, hidden_dim, rel_feat_dim)
    dbbc_fusion = DBBCFusion()
    
    # Load pretrained weights
    if not os.path.exists("dbbc_pretrained.pth"):
        print("dbbc_pretrained.pth not found. Please train the model first.")
        return
        
    dbbc_local.load_state_dict(torch.load("dbbc_pretrained.pth"))
    dbbc_local.eval()
    
    batch_size = 16
    num_strategic = 3 # N=3 strategic agents in our simulation
    seq_len = 5
    
    if override_csv is not None:
        latest_csv = override_csv
    else:
        csv_files = glob.glob("results/mcts_*.csv")
        if not csv_files:
            print("No CSV files found in results/ directory.")
            return
            
        latest_csv = max(csv_files, key=os.path.getctime)
    
    print(f"Loading CSV for evaluation: {latest_csv}")
    
    try:
        dataloader = get_dataloader(latest_csv, batch_size=batch_size, seq_len=seq_len)
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (obs_seqs, rel_feats, labels) in enumerate(dataloader):
            mus = []
            sigmas = []

            
            for i in range(num_strategic):
                agent_obs = obs_seqs[:, i, :, :]
                agent_rel = rel_feats[:, i, :]
                
                mu_i, sigma_i, _ = dbbc_local(agent_obs, agent_rel)
                mus.append(mu_i)
                sigmas.append(sigma_i)
                
            mus_tensor = torch.cat(mus, dim=-1)
            sigmas_tensor = torch.cat(sigmas, dim=-1)
            
            _, _, shared_beliefs = dbbc_fusion(mus_tensor, sigmas_tensor)
            
            # Predict adversary if belief > 0.5
            preds = (shared_beliefs > 0.5).float()
            
            all_preds.extend(preds.numpy().flatten())
            all_labels.extend(labels.numpy().flatten())
            
    if not all_preds:
        print("No evaluation data found or parsed.")
        return
        
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    print("\n--- DBBC Evaluation Results ---")
    print(f"Total evaluated windows: {len(all_labels)}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate DBBC Module")
    parser.add_argument('--csv', type=str, default=None, help='Specific CSV file to evaluate')
    args = parser.parse_args()
    
    evaluate(override_csv=args.csv)
