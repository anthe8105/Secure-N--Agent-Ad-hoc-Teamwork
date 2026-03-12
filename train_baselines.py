import torch
import numpy as np
import argparse
import os
from src.envs.LevelForagingEnv import load_default_scenario, LevelForagingEnv
from train_rapo import RAPOPPO
from src.models.dbbc import DBBCLocalEvidence, DBBCFusion

def train_baseline(scenario_id, baseline_name, max_episodes=50000, max_timesteps=100):
    print(f"\n{'='*50}")
    print(f"Starting Deep RL Training for: {baseline_name}")
    print(f"Scenario: {scenario_id} | Target Episodes: {max_episodes}")
    print(f"{'='*50}\n")
    
    # Environment Setup
    env, _ = load_default_scenario(method='mcts', scenario_id=scenario_id, display=False)
    
    # DBBC setup (Pretrained frozen weights for belief extraction)
    obs_dim = 15; action_dim_dbbc = 0; hidden_dim = 32; rel_feat_dim = 8
    dbbc_local = DBBCLocalEvidence(obs_dim, action_dim_dbbc, hidden_dim, rel_feat_dim)
    try:
        dbbc_local.load_state_dict(torch.load("dbbc_pretrained.pth"))
        print("[DBBC] Loaded pretrained weights successfully.")
    except:
        print("[DBBC] Warning: dbbc_pretrained.pth not found. Using random init.")
    dbbc_local.eval()
    
    # Initialize PPO Agent
    action_dim = 5
    
    # Baseline specific constraints
    risk_budget = 0.5
    if baseline_name in ["Vanilla", "Belief_Only", "Oracle"]:
        risk_budget = 999.0 # Effectively removes the CVaR penalty (unconstrained)
        
    rapo_agent = RAPOPPO(state_dim=40, action_dim=action_dim, 
                         lr_actor=0.0003, lr_critic=0.001, 
                         gamma=0.99, K_epochs=4, eps_clip=0.2, risk_budget=risk_budget)
                         
    update_timestep = max_timesteps * 4 # PPO update frequency
    time_step = 0
    
    # Create save directory
    os.makedirs("weights", exist_ok=True)
    save_path = f"weights/{baseline_name}_scenario{scenario_id}.pth"
    
    print("Training started (this will take a while)...")
    
    for episode in range(1, max_episodes+1):
        env.reset()
        ep_reward = 0
        ep_risk = 0
        
        # We need realistic inputs instead of dummies for actual training
        z_t = torch.randn(1, 32)
        
        # Ground truth mapping for Oracle
        ground_truth = {a.index: 1.0 if a.index.startswith('X') else 0.0 for a in env.components['agents']}

        for t in range(max_timesteps):
            time_step += 1
            
            # 1. Belief Extraction per baseline
            B_t = torch.zeros((1, 8))
            
            if baseline_name == "Vanilla":
                B_t = torch.zeros((1, 8))
            elif baseline_name == "Oracle":
                for i, gt in enumerate(ground_truth.values()):
                    if i < 8: B_t[0, i] = gt
            elif baseline_name in ["RAPO", "Belief_Only"]:
                # In full implementation, pass real history into DBBC here
                # B_t = dbbc_local(...)
                B_t = torch.rand((1, 8)) # placeholder for loop logic
            elif baseline_name == "Risk_Only":
                B_t = torch.zeros((1, 8))
                
            # 2. Action Selection
            action = rapo_agent.select_action(z_t, B_t)
            action_env = action.item() if torch.is_tensor(action) else int(action)
            
            # 3. Environment Step
            next_state, reward_env, done_env, info = env.step(action_env)
            
            reward = info.get('action reward', 0.0)
            rho = info.get('risk', 0.0)
            
            if baseline_name == "Vanilla":
                rho = 0.0
            elif baseline_name == "Risk_Only":
                ent = info.get('entanglements', {})
                rho = sum(ent.values())
                
            done = env.state_set.is_final_state(env)
            
            rapo_agent.buffer.rewards.append(reward)
            rapo_agent.buffer.risks.append(rho)
            rapo_agent.buffer.is_terminals.append(done)
            
            ep_reward += reward
            ep_risk += rho
            
            if time_step % update_timestep == 0:
                rapo_agent.update()
                
            if done:
                break
                
        # Logging
        if episode % 100 == 0:
            print(f"Ep {episode:5d} | Mean Ret: {ep_reward:6.2f} | Mean Risk: {ep_risk:6.2f} | Lambda: {rapo_agent.lagrangian_lambda:.4f}")
            
        # Checkpointing
        if episode % 1000 == 0:
            torch.save(rapo_agent.policy.state_dict(), save_path)
            
    # Final save
    torch.save(rapo_agent.policy.state_dict(), save_path)
    print(f"Training complete. Weights saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=int, default=10, help='Scenario ID')
    parser.add_argument('--baseline', type=str, required=True, choices=["Vanilla", "I-BAE", "Belief_Only", "Risk_Only", "RAPO", "Oracle"])
    parser.add_argument('--episodes', type=int, default=50000, help='Episodes to train')
    args = parser.parse_args()
    
    train_baseline(args.scenario, args.baseline, args.episodes)
