import torch
import numpy as np
from src.envs.LevelForagingEnv import load_default_scenario, LevelForagingEnv
from src.models.rapo import calculate_cvar
from train_rapo import RAPOPPO
import argparse

def evaluate_baseline(scenario_id, baseline_name, num_episodes=50):
    print(f"\nEvaluating configuration: {baseline_name} on Scenario {scenario_id}")
    
    # Load Environment
    env, _ = load_default_scenario(method='mcts', scenario_id=scenario_id, display=False)

    
    # Initialize PPO Agent specifically for evaluating this setup
    action_dim = 5
    rapo_agent = RAPOPPO(state_dim=40, action_dim=action_dim, 
                         lr_actor=0.0003, lr_critic=0.001, 
                         gamma=0.99, K_epochs=4, eps_clip=0.2, risk_budget=0.5)
                         
    # Attempt to load trained weights
    weight_path = f"weights/{baseline_name}_scenario{scenario_id}.pth"
    try:
        rapo_agent.policy.load_state_dict(torch.load(weight_path))
        print(f"[{baseline_name}] Successfully loaded trained weights.")
    except Exception as e:
        print(f"[{baseline_name}] No trained weights found at {weight_path}. Using random policy init.")
    
    rapo_agent.policy.eval()

    returns = []
    risks = []
    
    for ep in range(num_episodes):
        env.reset()
        ep_return = 0
        ep_risk = 0
        
        adhoc_agent = env.get_adhoc_agent()
        
        for t in range(50): # Max steps per episode
            # --- Baseline Configuration Logic ---
            # In a full run, we would query policies. Since we don't have trained weights,
            # we will simulate the behavior difference via mock flags and random outputs for skeleton testing.
            
            # Ground truth targets for Oracle
            ground_truth = {a.index: 1.0 if a.index.startswith('X') else 0.0 for a in env.components['agents']}
            
            B_t = torch.zeros((1, 8))
            use_cvar = False
            
            if baseline_name == "Vanilla":
                B_t = torch.zeros((1, 8)) # No belief
            elif baseline_name == "Oracle":
                # Provide true labels
                for i, gt in enumerate(ground_truth.values()):
                    if i < 8: B_t[0, i] = gt
                use_cvar = True
            elif baseline_name == "Belief_Only":
                # Simulated DBBC
                B_t = torch.tensor([[0.05]*8])
            elif baseline_name == "Risk_Only":
                B_t = torch.zeros((1, 8))
                use_cvar = True
            elif baseline_name == "RAPO":
                B_t = torch.tensor([[0.05]*8]) 
                use_cvar = True
                
            z_t = torch.randn(1, 32)
            # Agent takes action through policy network
            action_tensor = rapo_agent.select_action(z_t, B_t)
            action_env = action_tensor.item() if torch.is_tensor(action_tensor) else int(action_tensor)
            
            next_state, reward_env, done_env, info = env.step(action_env)
            r = info.get('action reward', 0.0)
            
            # Calculate Risk using local entanglements
            ent = info.get('entanglements', {})
            rho = 0.0
            
            if baseline_name == "Vanilla":
                rho = 0.0 # Vanilla ignores risk
            elif baseline_name == "Risk_Only":
                rho = sum(ent.values()) # Risk-only assumes all entanglements are dangerous
            else:
                # RAPO and others use DBBC Belief * Entanglement
                for j_idx_str, kappa in ent.items():
                    # Agents are 'A', 'B', 'C', 'X1'.. indices. We need a mapped index for B_t.
                    # Simplified for skeleton mapping:
                    try:
                        agent_idx_num = int(j_idx_str.replace('X','').replace('B','1').replace('C','2')) % 8
                        rho += B_t[0, agent_idx_num].item() * kappa
                    except:
                        rho += 0.05 * kappa
                        
            ep_return += r * (0.99 ** t) # Use discounted return
            ep_risk += rho
            
            if env.state_set.is_final_state(env):
                break
                
        returns.append(ep_return)
        risks.append(ep_risk)
        
    returns_tensor = torch.tensor(returns, dtype=torch.float32)
    risks_tensor = torch.tensor(risks, dtype=torch.float32)
    
    mean_return = returns_tensor.mean().item()
    cvar_risk = calculate_cvar(risks_tensor, beta=0.95).item()
    
    print(f"Results for {baseline_name}:")
    print(f"  Mean Return: {mean_return:.2f}")
    print(f"  Worst 5% CVaR Risk: {cvar_risk:.2f}")
    
    return mean_return, cvar_risk

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=int, default=10, help='Scenario ID (9-12 for Exps)')
    parser.add_argument('--episodes', type=int, default=10, help='Episodes per baseline')
    args = parser.parse_args()
    
    baselines = ["Vanilla", "I-BAE", "Belief_Only", "Risk_Only", "RAPO", "Oracle"]
    
    print("="*40)
    print(f"Starting Experimental Design Evaluation")
    print(f"Scenario: {args.scenario} | Episodes: {args.episodes}")
    print("="*40)
    
    # Calculate mock metrics
    for b in baselines:
        evaluate_baseline(args.scenario, b, args.episodes)
        
    print("\nEvaluation Script Complete.")
    print("To compute exact AUC-ROC and Brier scores, a trained DBBC model checkpoint must be loaded to generate predicted vs ground truth arrays over the test episodes.")
