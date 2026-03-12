import pandas as pd
import ast
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class AdhocSimulationDataset(Dataset):
    """
    Parses the AdLeap-MAS CSV log outputs into sequences for DBBC GRU training.
    """
    def __init__(self, csv_file, seq_len=5, num_strategic=3, K=2):
        self.seq_len = seq_len
        self.num_strategic = num_strategic
        self.K = K
        
        # Load the CSV
        print(f"Loading data from {csv_file}...")
        self.df = pd.read_csv(csv_file, sep=';', skipinitialspace=True)
        
        # Filter out corrupted rows or initial zeros
        self.df = self.df.dropna(subset=['TypeEstimation', 'ParametersEstimation'])
        
        self.samples = []
        self._process_data()
        
    def _process_data(self):
        """
        Converts string arrays from CSV into rolling window sequences.
        """
        raw_types = []
        raw_params = []
        
        for idx, row in self.df.iterrows():
            try:
                # The CSV saves lists as strings. Easiest way in python is to use literal_eval
                t_est = ast.literal_eval(row['TypeEstimation'])
                p_est = ast.literal_eval(row['ParametersEstimation'])
                
                raw_types.append(t_est)
                raw_params.append(p_est)
            except Exception as e:
                continue
                
        # We need a sequence length for the GRU
        for i in range(len(raw_types) - self.seq_len):
            # We select a target agent to track for this sequence window
            num_agents = len(raw_params[i])
            if num_agents == 0:
                continue
                
            target_idx = np.random.randint(0, num_agents)
            
            # Label Assignment: Assume the last K agents in the list are Adversaries
            is_adversary = 0.0
            if target_idx >= num_agents - self.K:
                is_adversary = 1.0
            ground_truth = torch.tensor([is_adversary], dtype=torch.float32)
            
            # We flatten the target agent's parameters to get 15 dims per step
            obs_seq_flat = []
            for t in range(i, i + self.seq_len):
                params_t = raw_params[t]
                target_params = params_t[target_idx] # list of 7 types, each with 3 params
                
                 # Flatten the list of lists
                flat_params = [item for sublist in target_params for item in sublist]
                
                # Take exactly 15 elements to match obs_dim + action_dim = 15
                if len(flat_params) >= 15:
                    flat_params = flat_params[:15]
                else:
                    flat_params = flat_params + [0.0] * (15 - len(flat_params))
                    
                obs_seq_flat.append(flat_params)
                
            obs_tensor = torch.tensor(obs_seq_flat, dtype=torch.float32)
            
            # Normalize observation sequence locally to promote scaling generalization (Min-Max)
            obs_min = obs_tensor.min(dim=0, keepdim=True)[0]
            obs_max = obs_tensor.max(dim=0, keepdim=True)[0]
            obs_range = obs_max - obs_min
            obs_range[obs_range == 0] = 1.0 # Prevent Division by 0
            obs_tensor_normalized = (obs_tensor - obs_min) / obs_range

            # Broadcast to mimic multiple strategic agents observing it: shape (num_strategic, seq_len, 15)
            obs_action_seqs = obs_tensor_normalized.unsqueeze(0).repeat(self.num_strategic, 1, 1)
            
            # 2. Construct Relational Features (num_strategic, 8)
            # We take the target agent's estimated type probabilities from the final step
            type_probs = raw_types[i + self.seq_len - 1][target_idx]
            # Ensure type_probs has exactly 7 elements before appending 0.0
            if len(type_probs) >= 7:
                type_probs = type_probs[:7]
            else:
                type_probs = type_probs + [0.0] * (7 - len(type_probs))
            
            # Pad with a 0.0 to reach the expected 8 dimensions
            rel_feat_list = type_probs + [0.0]
            rel_tensor = torch.tensor(rel_feat_list, dtype=torch.float32)
            # Broadcast to expected shape
            rel_features = rel_tensor.unsqueeze(0).repeat(self.num_strategic, 1)
            
            self.samples.append({
                'obs_seq': obs_action_seqs,
                'rel_feat': rel_features,
                'label': ground_truth
            })
            
        print(f"Processed {len(self.samples)} valid sequence windows.")
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['obs_seq'], sample['rel_feat'], sample['label']

def get_dataloader(csv_file, batch_size=16, seq_len=5):
    dataset = AdhocSimulationDataset(csv_file, seq_len=seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
