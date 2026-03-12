import torch
import torch.optim as optim
from src.models.dbbc import DBBCLocalEvidence, DBBCFusion, calculate_dbbc_losses
from src.data.dataset import get_dataloader
import numpy as np
import os
import glob
import argparse



def train(override_csv=None):
    # 1. Initialize DBBC Models
    obs_dim = 10
    action_dim = 5
    hidden_dim = 32
    rel_feat_dim = 8
    
    # We instantiate one DBBC Local Evidence module (parameters are shared across strategic agents)
    dbbc_local = DBBCLocalEvidence(obs_dim, action_dim, hidden_dim, rel_feat_dim)
    
    # We instantiate the fusion module
    dbbc_fusion = DBBCFusion()
    
    # Define optimizer
    optimizer = optim.Adam(dbbc_local.parameters(), lr=1e-3)
    
    epochs = 100
    batch_size = 16
    num_strategic = 3 # N=3 strategic agents in our simulation
    seq_len = 5
    
    if override_csv is not None:
        latest_csv = override_csv
    else:
        # Dynamically find the newest output CSV from the collect_data loop
        csv_files = glob.glob("results/mcts_*.csv")
        if not csv_files:
            print("Waiting for collect_data.py to generate CSVs...")
            return
            
        latest_csv = max(csv_files, key=os.path.getctime)
    
    print(f"Loading CSV for training: {latest_csv}")
    
    # 1. Load Real Data from the AdLeap logger!
    print("Starting DBBC Pre-training on Data...")
    
    dataloader = get_dataloader(latest_csv, batch_size=batch_size, seq_len=seq_len)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        
        for batch_idx, (obs_seqs, rel_feats, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Print tensor shapes once just so we can verify the pipeline is working
            if epoch == 0 and batch_idx == 0:
                print(f"Tracking Shape: Obs={obs_seqs.shape}, Rel={rel_feats.shape}, Labels={labels.shape}")
            
            all_mus = []
            all_sigmas = []
            
            # 2. Compute local evidence for each strategic agent i
            # DataLoader returns batch_first: [batch, num_strategic, seq_len, obs_dim+action_dim]
            for i in range(num_strategic):
                agent_obs = obs_seqs[:, i, :, :]   # Shape: (batch, seq_len, 15)
                agent_rel = rel_feats[:, i, :]     # Shape: (batch, 8)
                
                mu_i, sigma_i, local_belief_i = dbbc_local(agent_obs, agent_rel)
                all_mus.append(mu_i)
                all_sigmas.append(sigma_i)
                
            # Concatenate outputs: shape (batch, num_strategic)
            mus_tensor = torch.cat(all_mus, dim=-1)
            sigmas_tensor = torch.cat(all_sigmas, dim=-1)
            
            # 3. Fuse the evidence
            fused_mu, fused_sigma, shared_beliefs = dbbc_fusion(mus_tensor, sigmas_tensor)
            
            # 4. Calculate Losses
            l_belief, l_det, l_cal = calculate_dbbc_losses(shared_beliefs, labels, lambda_cal=0.1)
            
            # 5. Backpropagation
            l_belief.backward()
            optimizer.step()
            
            epoch_loss += l_belief.item()
            batches += 1
            
        if batches > 0:
            print(f"Epoch [{epoch+1}/{epochs}], Average Total Loss: {epoch_loss/batches:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{epochs}], No data batches found.")

    print("Finished Pre-training DBBC Module.")
    
    # Save the model
    torch.save(dbbc_local.state_dict(), "dbbc_pretrained.pth")
    print("Model saved to dbbc_pretrained.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DBBC Module")
    parser.add_argument('--csv', type=str, default=None, help='Specific CSV file to train on')
    args = parser.parse_args()
    
    train(override_csv=args.csv)
