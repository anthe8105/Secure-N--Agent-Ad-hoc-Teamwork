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


# ---------------------------------------------------------------------------
# DBBCDetector: simplified single-observer variant for state-based training
# ---------------------------------------------------------------------------

class DBBCDetector(nn.Module):
    """
    Simplified DBBC detector for the small 3-agent scenario (N=1 strategic).

    Aligned with paper §3.1:
      - Recurrent encoder Enc_phi: GRU over h_t = (o_0,a_0,...,o_t,a_t)
      - MLP head: z_t → b_t(j) ∈ [0,1]  (adversary belief for agent j)
      - Loss: BCE(b,y) + lambda_cal·MSE(b,y)  (paper Eq. 8)

    No PoE fusion is needed here because N=1 (single strategic observer).
    Relative positions are encoded inside the observation vector, so no
    separate relational feature branch is required.

    Args:
        input_dim:  obs_dim + action_dim (default 11+5=16)
        hidden_dim: GRU hidden size (default 32)
    """
    def __init__(self, input_dim=16, hidden_dim=32):
        super(DBBCDetector, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Enc_phi: recurrent encoder over (obs, action) history
        self.encoder = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)

        # MLP head: z_t → belief b_t(j)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, seq):
        """
        seq: (batch, seq_len, input_dim)  — concatenated (obs, action) history

        Returns:
            belief: (batch, 1) — probability that the observed agent is an adversary
        """
        _, hidden = self.encoder(seq)          # hidden: (1, batch, hidden_dim)
        z_t = hidden.squeeze(0)                # (batch, hidden_dim)

        x = F.relu(self.fc1(z_t))
        belief = torch.sigmoid(self.fc2(x))    # (batch, 1)
        return belief


# ---------------------------------------------------------------------------
# DBBCMultiObserver: full DBBC with N>1 strategic agents + PoE fusion
# ---------------------------------------------------------------------------

class DBBCMultiObserver(nn.Module):
    """
    Full DBBC for N strategic agents observing uncontrolled agents (paper §3.1).

    For each uncontrolled target agent j:
      1. Each strategic agent i runs DBBCLocalEvidence(h_t^{ij}, ψ_t^{ij})
         → (μ_{ij,t}, σ_{ij,t}, local_belief_{ij,t})
      2. DBBCFusion merges across all i (precision-weighted PoE):
         → (μ̃_t(j), σ̃_t(j), shared_belief_t(j))
      3. Loss: BCE(b̃, y_j) + λ·MSE(b̃, y_j)  per agent j  (Eq. 8)

    This wrapper creates N independent DBBCLocalEvidence modules (one per
    strategic observer) and a single DBBCFusion module.

    Args:
        n_observers:   Number of strategic agents (N)
        input_dim:     obs_dim + action_dim  (e.g. 13+5=18 for Scenario 14)
        hidden_dim:    GRU hidden size
        rel_feat_dim:  Relational feature dimension (default 3)
    """
    def __init__(self, n_observers, input_dim=18, hidden_dim=64, rel_feat_dim=3):
        super(DBBCMultiObserver, self).__init__()
        self.n_observers = n_observers

        # One local evidence encoder per strategic observer
        self.encoders = nn.ModuleList([
            DBBCLocalEvidence(
                obs_dim=input_dim,      # obs_action_dim (GRU input)
                action_dim=0,           # action already concatenated into input_dim
                hidden_dim=hidden_dim,
                rel_feat_dim=rel_feat_dim
            )
            for _ in range(n_observers)
        ])
        self.fusion = DBBCFusion()

    def forward(self, seqs, rel_features):
        """
        Args:
            seqs:         (batch, N, seq_len, input_dim)
            rel_features: (batch, N, rel_feat_dim)

        Returns:
            shared_belief: (batch, 1) — fused adversary probability
            fused_mu:      (batch, 1) — fused logit
            fused_sigma:   (batch, 1) — fused uncertainty
        """
        mus = []
        sigmas = []
        for i, encoder in enumerate(self.encoders):
            seq_i = seqs[:, i, :, :]            # (batch, seq_len, input_dim)
            rel_i = rel_features[:, i, :]       # (batch, rel_feat_dim)
            mu_i, sigma_i, _ = encoder(seq_i, rel_i)
            mus.append(mu_i)                    # (batch, 1)
            sigmas.append(sigma_i)              # (batch, 1)

        # Stack along observer dimension: (batch, N)
        mus    = torch.cat(mus,    dim=-1)      # (batch, N)
        sigmas = torch.cat(sigmas, dim=-1)      # (batch, N)

        # Precision-weighted fusion (paper Eq. 3-5)
        fused_mu, fused_sigma, shared_belief = self.fusion(mus, sigmas)

        return shared_belief, fused_mu, fused_sigma

