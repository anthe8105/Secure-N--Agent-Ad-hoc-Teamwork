import torch
import torch.nn as nn
import torch.nn.functional as F

class RAPOActorCritic(nn.Module):
    """
    Module 2: RAPO (Risk-Aware Policy Optimization)
    This module implements the decentralized policy and value networks.
    It conditions on the agent's history (via DBBC's GRU hidden state) 
    and the shared belief map B_t to optimize return while controlling risk.
    """
    def __init__(self, hidden_dim, belief_dim, action_dim):
        super(RAPOActorCritic, self).__init__()
        
        # We assume the history is encoded by the DBBC module into `hidden_dim`
        # and the belief map provides `belief_dim` features (e.g., probability of each teammate being an adversary).
        
        input_dim = hidden_dim + belief_dim
        
        # --- Actor Network (pi_theta) ---
        self.actor_fc1 = nn.Linear(input_dim, 64)
        self.actor_fc2 = nn.Linear(64, 64)
        self.actor_out = nn.Linear(64, action_dim)
        
        # --- Critic Network (V_theta) ---
        self.critic_fc1 = nn.Linear(input_dim, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_out = nn.Linear(64, 1)
        
    def forward(self, z_t, B_t):
        """
        z_t: Hidden state from DBBC's GRU encoder representing history h_t^i. Shape: (batch, hidden_dim)
        B_t: The shared belief map over uncontrolled agents. Shape: (batch, belief_dim)
        """
        # Concatenate history and belief
        x = torch.cat([z_t, B_t], dim=-1)
        
        # Actor
        a = F.relu(self.actor_fc1(x))
        a = F.relu(self.actor_fc2(a))
        action_logits = self.actor_out(a)
        
        # Critic
        v = F.relu(self.critic_fc1(x))
        v = F.relu(self.critic_fc2(v))
        state_value = self.critic_out(v)
        
        return action_logits, state_value

def calculate_cvar(risks, beta=0.95):
    """
    Calculates the Conditional Value-at-Risk (Upper-Tail CVaR) 
    using the Rockafellar-Uryasev representation.
    
    risks: A tensor of discounted trajectory risks R(tau). Shape: (batch_size,)
    beta: The confidence level (e.g., 0.95 focuses on the worst 5%).
    """
    # Sort the risks in ascending order
    sorted_risks, _ = torch.sort(risks)
    
    # Calculate the Value-at-Risk (VaR) index
    var_idx = int(beta * len(sorted_risks))
    
    # If the batch is too small, just return the max risk
    if var_idx >= len(sorted_risks):
        return sorted_risks[-1]
        
    # The tail is everything above the VaR index
    tail_risks = sorted_risks[var_idx:]
    
    # CVaR is the expectation over the tail
    cvar = torch.mean(tail_risks)
    return cvar
