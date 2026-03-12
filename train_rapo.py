import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.risks = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.risks[:]
        del self.state_values[:]
        del self.is_terminals[:]

class RAPOPPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, risk_budget):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()
        
        # Dual variable for the Lagrangian relaxation
        self.lagrangian_lambda = 0.0
        self.risk_budget = risk_budget
        self.eta_lambda = 0.1 # learning rate for dual variable update
        
        # Import the model from the newly created rapo module
        from src.models.rapo import RAPOActorCritic
        # We assume state_dim is the concatenated size of (z_t, B_t)
        # For a standard interface, we'll initialize the network. 
        # (Note: In full CTDE, hidden_dim and belief_dim would be passed cleanly).
        # We assume 32 hidden + 8 belief = 40 dim for this wrapper
        self.policy = RAPOActorCritic(hidden_dim=32, belief_dim=8, action_dim=action_dim)
        self.optimizer = optim.Adam([
            {'params': self.policy.actor_fc1.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor_fc2.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor_out.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic_fc1.parameters(), 'lr': lr_critic},
            {'params': self.policy.critic_fc2.parameters(), 'lr': lr_critic},
            {'params': self.policy.critic_out.parameters(), 'lr': lr_critic}
        ])
        
        self.policy_old = RAPOActorCritic(hidden_dim=32, belief_dim=8, action_dim=action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, z_t, B_t):
        with torch.no_grad():
            action_logits, state_val = self.policy_old(z_t, B_t)
            action_probs = F.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)
            
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            
        self.buffer.states.append(torch.cat([z_t, B_t], dim=-1))
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()

    def update(self):
        from src.models.rapo import calculate_cvar
        
        # Monte Carlo estimate of cumulative rewards and risks
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        risks = []
        discounted_risk = 0
        for risk, is_terminal in zip(reversed(self.buffer.risks), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_risk = 0
            discounted_risk = risk + (self.gamma * discounted_risk)
            risks.insert(0, discounted_risk)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Calculate CVaR metrics (Algorithm 1: Sample-based CVaR estimation)
        risks_tensor = torch.tensor(risks, dtype=torch.float32)
        cvar_risk = calculate_cvar(risks_tensor, beta=0.95)
        
        # convert list to tensor and remove the extra dimension of size 1
        old_states = torch.cat(self.buffer.states).detach()
        old_actions = torch.tensor(self.buffer.actions).detach()
        old_logprobs = torch.tensor(self.buffer.logprobs).detach()
        old_state_values = torch.cat(self.buffer.state_values).detach()

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach().squeeze()

        for _ in range(self.K_epochs):
            # Split state into z_t and B_t for the forward pass
            z_t = old_states[:, :32] 
            B_t = old_states[:, 32:]
            
            action_logits, state_values = self.policy(z_t, B_t)
            action_probs = F.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)

            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # L_PPO = Return advantage
            loss_ppo = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values.squeeze(), rewards) - 0.01 * dist_entropy
            
            # L(theta, lambda) = L_PPO - lambda * (CVaR - delta)  [We subtract the penalty to maximize the objective]
            # Since PyTorch minimizes loss, we add the Lagrangian penalty to the loss function
            loss_total = loss_ppo.mean() + self.lagrangian_lambda * (cvar_risk - self.risk_budget)
            
            # take gradient step
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        
        # Update Lagrangian Dual Variable via projected ascent
        # lambda <- max(0, lambda + eta_lambda * (CVaR - risk_budget))
        cvar_violation = (cvar_risk - self.risk_budget).item()
        self.lagrangian_lambda = max(0.0, self.lagrangian_lambda + self.eta_lambda * cvar_violation)
        print(f"Updated Lambda: {self.lagrangian_lambda:.4f} | Measured CVaR Risk: {cvar_risk.item():.4f}")
