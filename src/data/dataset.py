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


# ---------------------------------------------------------------------------
# New dataset for raw (state, action) sequences — aligned with paper §3.1
# ---------------------------------------------------------------------------

class StateActionSequenceDataset(Dataset):
    """
    Reads raw (observation, action) data produced by collect_state_data.py.

    CSV columns: Step; AgentIndex; IsAdversary; StateVec; Action
    - StateVec: string-encoded Python list of floats (obs_dim = 10)
    - Action:   integer 0-4, converted to 1-hot (action_dim = 5)
    - IsAdversary: 0 or 1

    Builds rolling windows of length `seq_len` (default H=10) per agent,
    matching the GRU input h_t = (o_0,a_0,...,o_t,a_t) from the paper.

    Temporal split support
    ----------------------
    To avoid data leakage (adjacent windows share H-1 steps), we split at
    the STEP level, not the window level:
      split='train': steps  0 .. floor(n * train_ratio) - 1
      split='test':  steps  floor(n * train_ratio) + seq_len .. n-1
                     (seq_len gap prevents any train step appearing in a
                      test window)
      split=None:    use all steps (backward-compatible default)

    Returns: (seq_tensor, label)
      seq_tensor: (seq_len, obs_dim + action_dim)   dtype=float32
      label:      (1,)                               dtype=float32
    """
    OBS_DIM = 10
    ACTION_DIM = 5

    def __init__(self, csv_file, seq_len=10, split=None, train_ratio=0.7):
        """
        Args:
            csv_file:    Path to state_data CSV.
            seq_len:     History window length H (default 10).
            split:       None | 'train' | 'test'
            train_ratio: Fraction of steps for training (default 0.7).
        """
        self.seq_len = seq_len
        self.input_dim = self.OBS_DIM + self.ACTION_DIM
        assert split in (None, 'train', 'test'), \
            f"split must be None, 'train' or 'test', got {split!r}"

        df = pd.read_csv(csv_file, sep=';', skipinitialspace=True)
        df = df.dropna(subset=['StateVec', 'Action', 'IsAdversary'])
        df['Step'] = df['Step'].astype(int)
        df = df.sort_values(['AgentIndex', 'Step']).reset_index(drop=True)
        n_agents = df['AgentIndex'].nunique()

        self.samples = []
        for agent_id, group in df.groupby('AgentIndex'):
            group = group.reset_index(drop=True)
            is_adv = float(group['IsAdversary'].iloc[0])

            obs_list, act_list = [], []
            for _, row in group.iterrows():
                try:
                    obs_vec = ast.literal_eval(row['StateVec'])
                    obs_vec = (obs_vec + [0.0] * self.OBS_DIM)[:self.OBS_DIM]
                    action_idx = int(row['Action'])
                    one_hot = [0.0] * self.ACTION_DIM
                    if 0 <= action_idx < self.ACTION_DIM:
                        one_hot[action_idx] = 1.0
                    obs_list.append(obs_vec)
                    act_list.append(one_hot)
                except Exception:
                    continue

            n = len(obs_list)
            if n < seq_len:
                continue

            # Temporal boundary
            cut = int(n * train_ratio)
            if split is None:
                start, end = 0, n
            elif split == 'train':
                start, end = 0, cut
            else:  # 'test' — leave a gap of seq_len to prevent overlap leakage
                start, end = cut + seq_len, n

            for i in range(start, end - seq_len + 1):
                combined = [obs_list[i+k] + act_list[i+k] for k in range(seq_len)]
                seq_tensor = torch.tensor(combined, dtype=torch.float32)
                label = torch.tensor([is_adv], dtype=torch.float32)
                self.samples.append((seq_tensor, label))

        tag = f"[{split}]" if split else "[all]"
        gap = seq_len if split == 'test' else 0
        print(f"[StateActionSequenceDataset]{tag} "
              f"{len(self.samples)} windows from {n_agents} agents "
              f"(train_ratio={train_ratio}, gap={gap} steps)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def get_state_action_dataloader(csv_file, batch_size=32, seq_len=10):
    """Backward-compatible: uses all data (no split)."""
    dataset = StateActionSequenceDataset(csv_file, seq_len=seq_len, split=None)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def get_train_test_dataloaders(csv_file, batch_size=32, seq_len=10, train_ratio=0.7):
    """
    Returns (train_loader, test_loader) with a leak-free temporal split.

    For each agent's n steps:
        Train windows: steps [0, floor(n*r))
        Gap:           seq_len steps (no windows start here)
        Test windows:  steps [floor(n*r) + seq_len, n)

    The gap of H steps ensures no training observation ever appears inside
    any test window (since each window spans H consecutive steps).
    """
    train_ds = StateActionSequenceDataset(
        csv_file, seq_len=seq_len, split='train', train_ratio=train_ratio)
    test_ds = StateActionSequenceDataset(
        csv_file, seq_len=seq_len, split='test',  train_ratio=train_ratio)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, drop_last=False)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Multi-observer dataset for N>1 strategic agents — uses DBBCLocalEvidence
# ---------------------------------------------------------------------------

class MultiObserverDataset(Dataset):
    """
    Reads multi-observer (state, action) data from collect_multi_data.py.

    CSV columns: Step; ObserverId; AgentIndex; IsAdversary; StateVec; Action; RelFeatures

    For each (observer_i, target_j) pair, builds rolling windows of H steps.
    Each sample is a dict with ALL N observers' views of the SAME target j
    at the SAME time window, so that the fusion module can combine them.

    Returns per sample:
      {
        'seqs':         Tensor (N_observers, seq_len, input_dim),
        'rel_features': Tensor (N_observers, rel_feat_dim),
        'label':        Tensor (1,)  — 0.0 or 1.0
      }

    Temporal split:
      split='train': first train_ratio of steps per (observer, target) pair
      split='test':  last (1-train_ratio) with seq_len gap
      split=None:    all steps
    """
    ACTION_DIM = 5
    REL_FEAT_DIM = 3

    def __init__(self, csv_file, seq_len=10, split=None, train_ratio=0.7):
        self.seq_len = seq_len
        assert split in (None, 'train', 'test')

        df = pd.read_csv(csv_file, sep=';', skipinitialspace=True)
        df = df.dropna(subset=['StateVec', 'Action', 'IsAdversary'])
        df['Step'] = df['Step'].astype(int)

        # Discover dimensions from first row
        first_obs = ast.literal_eval(df['StateVec'].iloc[0])
        self.obs_dim = len(first_obs)
        self.input_dim = self.obs_dim + self.ACTION_DIM

        # Identify observers and targets
        observers = sorted(df['ObserverId'].unique())
        targets = sorted(df['AgentIndex'].unique())
        self.n_observers = len(observers)

        # Pre-parse all (observer, target) timeseries
        # key = (observer_id, target_id) → list of (obs_vec, action_onehot, rel_feat, step)
        pair_data = {}
        for (obs_id, tgt_id), group in df.groupby(['ObserverId', 'AgentIndex']):
            group = group.sort_values('Step').reset_index(drop=True)
            is_adv = float(group['IsAdversary'].iloc[0])
            entries = []
            for _, row in group.iterrows():
                try:
                    obs_vec = ast.literal_eval(row['StateVec'])
                    obs_vec = (obs_vec + [0.0] * self.obs_dim)[:self.obs_dim]
                    action_idx = int(row['Action'])
                    one_hot = [0.0] * self.ACTION_DIM
                    if 0 <= action_idx < self.ACTION_DIM:
                        one_hot[action_idx] = 1.0
                    rel = ast.literal_eval(row['RelFeatures']) if 'RelFeatures' in row.index else [0.0]*self.REL_FEAT_DIM
                    rel = (rel + [0.0]*self.REL_FEAT_DIM)[:self.REL_FEAT_DIM]
                    entries.append((obs_vec, one_hot, rel))
                except Exception:
                    continue
            pair_data[(obs_id, tgt_id)] = (entries, is_adv)

        # Build aligned windows: for each target j, align across all observers
        self.samples = []
        for tgt_id in targets:
            # Find the minimum length across all observers for this target
            lengths = []
            for obs_id in observers:
                key = (obs_id, tgt_id)
                if key not in pair_data:
                    break
                lengths.append(len(pair_data[key][0]))
            else:
                # All observers have data for this target
                n = min(lengths)
                if n < seq_len:
                    continue
                is_adv = pair_data[(observers[0], tgt_id)][1]

                # Temporal boundary
                cut = int(n * train_ratio)
                if split is None:
                    start, end = 0, n
                elif split == 'train':
                    start, end = 0, cut
                else:
                    start, end = cut + seq_len, n

                for i in range(start, end - seq_len + 1):
                    # For each observer, extract the window [i, i+seq_len)
                    obs_seqs = []
                    obs_rels = []
                    for obs_id in observers:
                        entries = pair_data[(obs_id, tgt_id)][0]
                        combined = [entries[i+k][0] + entries[i+k][1] for k in range(seq_len)]
                        obs_seqs.append(torch.tensor(combined, dtype=torch.float32))
                        # Use relational features from the LAST step in the window
                        obs_rels.append(torch.tensor(entries[i+seq_len-1][2], dtype=torch.float32))

                    sample = {
                        'seqs': torch.stack(obs_seqs),           # (N, seq_len, input_dim)
                        'rel_features': torch.stack(obs_rels),   # (N, rel_feat_dim)
                        'label': torch.tensor([is_adv], dtype=torch.float32),  # (1,)
                    }
                    self.samples.append(sample)

        tag = f"[{split}]" if split else "[all]"
        gap = seq_len if split == 'test' else 0
        print(f"[MultiObserverDataset]{tag} "
              f"{len(self.samples)} windows, {self.n_observers} observers, "
              f"{len(targets)} targets (obs_dim={self.obs_dim}, input_dim={self.input_dim}, "
              f"gap={gap})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _multi_collate_fn(batch):
    """Custom collate for MultiObserverDataset dicts."""
    seqs = torch.stack([s['seqs'] for s in batch])                # (B, N, H, D)
    rel_features = torch.stack([s['rel_features'] for s in batch]) # (B, N, R)
    labels = torch.stack([s['label'] for s in batch])              # (B, 1)
    return {'seqs': seqs, 'rel_features': rel_features, 'label': labels}


def get_multi_train_test_dataloaders(csv_file, batch_size=32, seq_len=10, train_ratio=0.7):
    """
    Returns (train_loader, test_loader) for multi-observer detection.
    Each batch is a dict: {seqs: (B,N,H,D), rel_features: (B,N,R), label: (B,1)}.
    """
    train_ds = MultiObserverDataset(csv_file, seq_len=seq_len, split='train', train_ratio=train_ratio)
    test_ds  = MultiObserverDataset(csv_file, seq_len=seq_len, split='test',  train_ratio=train_ratio)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True, collate_fn=_multi_collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, drop_last=False, collate_fn=_multi_collate_fn)
    return train_loader, test_loader


