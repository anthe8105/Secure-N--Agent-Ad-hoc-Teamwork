import torch
import torch.nn as nn
import torch.nn.functional as F

class DBBCLocalEvidence(nn.Module):
    """
    Module 1: DBBC (Distributed Bayesian Belief Consensus)
    This module computes the local evidence for each agent (Section 3.1.1).
    It uses a recurrent encoder for the agent's history and a linear head
    for relational features to output a mean and variance (uncertainty) 
    over the adversariality logit for a target uncontrolled agent.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim, rel_feat_dim):
        super(DBBCLocalEvidence, self).__init__()
        
        # Recurrent encoder (Enc_phi): (O x A)^{t+1} -> R^d
        self.encoder = nn.GRU(input_size=obs_dim + action_dim, hidden_size=hidden_dim, batch_first=True)
        
        # Neural head f_phi: (z^i_t, psi^{ij}_t) -> (mu_{ij,t}, sigma_{ij,t})
        # Concatenate hidden state and relational features
        self.fc1 = nn.Linear(hidden_dim + rel_feat_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        
        # Outputs mean and log_std (for numerical stability)
        self.mu_head = nn.Linear(32, 1)
        self.log_std_head = nn.Linear(32, 1)
        
    def forward(self, obs_action_seq, rel_features):
        """
        obs_action_seq: Sequence of (observation, action) up to time t. Shape: (batch, seq_len, obs_dim + action_dim)
        rel_features: Relational features psi^{ij}_t. Shape: (batch, rel_feat_dim)
        """
        # Encode history
        # out shape: (batch, seq_len, hidden_dim)
        # hidden shape: (1, batch, hidden_dim)
        out, hidden = self.encoder(obs_action_seq)
        
        # Use the final hidden state z^i_t
        z_t = hidden.squeeze(0) # (batch, hidden_dim)
        
        # Combine with relational features
        x = torch.cat([z_t, rel_features], dim=-1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        mu = self.mu_head(x)          # (batch, 1)
        log_std = self.log_std_head(x) # (batch, 1)
        
        # Enforce sigma > 0 using softplus
        sigma = F.softplus(log_std) + 1e-6 
        
        # The local belief plugin approximation is sigm(mu)
        local_belief = torch.sigmoid(mu)
        
        return mu, sigma, local_belief


class DBBCFusion(nn.Module):
    """
    Module 1: Trusted Bayesian Fusion (Section 3.1.2)
    Precision-weighted product-of-experts in logit space.
    """
    def __init__(self, epsilon=1e-5):
        super(DBBCFusion, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, mus, sigmas):
        """
        mus: Means from all strategic agents for target j. Shape: (batch, num_strategic)
        sigmas: Variances (or stds) from all strategic agents. Shape: (batch, num_strategic)
        """
        # Calculate precision weights: w_k = 1 / (sigma_k^2 + epsilon)
        variances = sigmas ** 2
        weights = 1.0 / (variances + self.epsilon)
        
        # Sum of weights
        sum_weights = torch.sum(weights, dim=-1, keepdim=True)
        
        # Fused logit mean (Equation 3): (sum_k w_k * mu_k) / sum_k w_k
        weighted_mu_sum = torch.sum(weights * mus, dim=-1, keepdim=True)
        fused_mu = weighted_mu_sum / sum_weights
        
        # Fused logit variance (Equation 4): (sum_k w_k)^-1
        fused_var = 1.0 / sum_weights
        fused_sigma = torch.sqrt(fused_var)
        
        # Shared belief map (Equation 5)
        shared_belief = torch.sigmoid(fused_mu)
        
        return fused_mu, fused_sigma, shared_belief


def calculate_dbbc_losses(shared_beliefs, ground_truths, lambda_cal=0.1):
    """
    Calculates the training losses for the DBBC module (Section 3.1.4).
    
    shared_beliefs: The fused posterior probability b_t(j). Shape: (batch, 1)
    ground_truths: The actual type y_j in {0, 1}. Shape: (batch, 1)
    lambda_cal: Weight for the calibration loss.
    """
    # Detection loss (Equation 6) - Binary Cross Entropy
    l_det = F.binary_cross_entropy(shared_beliefs, ground_truths, reduction='sum')
    
    # Calibration loss (Equation 7) - Brier Score / MSE
    l_cal = F.mse_loss(shared_beliefs, ground_truths, reduction='sum')
    
    # Total belief loss (Equation 8)
    l_belief = l_det + lambda_cal * l_cal
    
    return l_belief, l_det, l_cal
