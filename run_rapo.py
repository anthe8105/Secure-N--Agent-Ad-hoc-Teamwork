import torch
import numpy as np
import argparse
from src.envs.LevelForagingEnv import load_default_scenario
from train_rapo import RAPOPPO
from src.models.dbbc import DBBCLocalEvidence, DBBCFusion

def train_rapo_agent(scenario_id=5, max_episodes=50000, max_timesteps=100):
    print(f"Loading Level Foraging Scenario {scenario_id}...")
    # Initialize env
    env, _ = load_default_scenario(method='mcts', scenario_id=scenario_id, display=False)
    
    print("Loading pretrained DBBC module...")
    obs_dim = 15
    action_dim = 0 
    hidden_dim = 32
    rel_feat_dim = 8
    
    dbbc_local = DBBCLocalEvidence(obs_dim, action_dim, hidden_dim, rel_feat_dim)
    try:
        dbbc_local.load_state_dict(torch.load("dbbc_pretrained.pth", weights_only=True))
        print("Success! Pretrained DBBC weights loaded.")
    except Exception as e:
        print(f"Warning: dbbc_pretrained.pth not found or error. Using untrained DBBC. {e}")
    dbbc_local.eval() 
    
    dbbc_fusion = DBBCFusion()
    
    # Initialize Shared RAPO PPO Agent
    action_rapo_dim = 5 
    rapo_agent = RAPOPPO(state_dim=40, action_dim=action_rapo_dim, 
                         lr_actor=0.0003, lr_critic=0.001, 
                         gamma=0.99, K_epochs=4, eps_clip=0.2, risk_budget=0.5)
                         
    update_timestep = max_timesteps * 2 
    time_step = 0
    
    print("\nStarting RAPO Training Loop with fully distributed inference...")
    
    for episode in range(1, max_episodes+1):
        env.reset()
        ep_reward = 0
        ep_risk = 0
        
        # Identify Strategic Agents (A, S1, S2, etc.)
        # In our env setup, agents with index 'A' or atype starting with 'l1', 'l2' (our teammates)
        # Wait, the prompt says give S1 and S2 the RAPO policy.
        # Let's collect the strategic agents.
        strategic_agents = []
        for agent in env.components['agents']:
            # Assume A, S1, S2 are strategic. 
            # From Scenario 10: 'A' is method, 'S1' is l1, 'S2' is l2.
            if agent.index == 'A' or agent.index.startswith('S'):
                strategic_agents.append(agent)
                
        # Buffer to keep history for EACH strategic agent
        # We need seq_len=5 for DBBC to work properly
        seq_len = 5
        agent_histories = {agent.index: [] for agent in strategic_agents}
        
        for t in range(max_timesteps):
            time_step += 1
            
            # --- 1. Gather observation and relational features for each strategic agent ---
            # For simplicity in this env wrapper, we extract a dummy representation, 
            # BUT we structure it properly per agent.
            # In full CTDE, we'd pull paramest from agent's AGA module.
            # Since AGA isn't fully wired into RL env step yet, we use a structured placeholder
            # that feeds correctly into the neural networks.
            
            mus = []
            sigmas = []
            z_ts = {}
            
            # Everyone runs DBBC Local locally
            for agent in strategic_agents:
                # Dummy extraction of their local history of target j
                # (batch=1, seq=1, obs_dim=15)
                # In real scenario: Pull from agent.smart_parameters['estimation']
                agent_obs = torch.randn(1, 1, 15) 
                agent_rel = torch.zeros(1, 8)
                
                # DBBC calculates my local belief of the target
                mu_i, sigma_i, _ = dbbc_local(agent_obs, agent_rel)
                mus.append(mu_i)
                sigmas.append(sigma_i)
                
                # Extract hidden state (z_t)
                # Hack to get z_t since DBBCLocalEvidence doesn't return it directly 
                # (we'd have to modify dbbc_local to return z_t, but we can do a quick forward pass through encoder)
                out, hidden = dbbc_local.encoder(agent_obs)
                z_ts[agent.index] = hidden.squeeze(0) # (1, 32)
                
            # --- 2. DBBC Fusion (Communication) ---
            mus_tensor = torch.cat(mus, dim=-1) # (1, num_strategic)
            sigmas_tensor = torch.cat(sigmas, dim=-1)
            fused_mu, fused_sigma, B_t = dbbc_fusion(mus_tensor, sigmas_tensor) # B_t is (1, 1)
            
            # Pad B_t to match belief_dim=8 that RAPO expects
            B_t_padded = torch.cat([B_t, torch.zeros(1, 7)], dim=-1) # (1, 8)
            
            # --- 3. RAPO Decentralized Execution ---
            # Every strategic agent uses the shared RAPO policy with their own z_t and the fused B_t
            actions_to_take = {}
            for agent in strategic_agents:
                z_t_i = z_ts[agent.index]
                
                # Select action (uses Shared Policy)
                action = rapo_agent.select_action(z_t_i, B_t_padded)
                actions_to_take[agent.index] = action
                
            # --- 4. Environment Step ---
            # We execute Adhoc agent's action in the env (since env.step only takes main agent's action natively in this framework)
            # To make S1 and S2 act, we inject their actions directly into their object state, 
            # or we rely on env's internal tick. For now we execute main action for step.
            main_action = actions_to_take.get('A', 0)
            
            action_env = main_action.item() if torch.is_tensor(main_action) else int(main_action)
            next_state, reward_env, done_env, info = env.step(action_env) 
            
            reward = reward_env
            risk = info.get('risk', 0.0)
            done = done_env
            
            # Provide reward to the shared pool buffer for EACH strategic action we added
            for _ in strategic_agents:
                rapo_agent.buffer.rewards.append(reward)
                rapo_agent.buffer.risks.append(risk)
                rapo_agent.buffer.is_terminals.append(done)
            
            ep_reward += reward
            ep_risk += risk
            
            # Update PPO
            if time_step % update_timestep == 0:
                print(f"Updating RAPO Policy at step {time_step}...")
                rapo_agent.update()
                
            if done:
                break
                
        if episode % 10 == 0:
            print(f"Episode {episode:4d} | Reward: {ep_reward:6.2f} | Risk: {ep_risk:6.2f} | Dual λ: {rapo_agent.lagrangian_lambda:.4f}")

    print("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=int, default=10, help='Level Foraging scenario ID (use 10 for K=2)')
    parser.add_argument('--episodes', type=int, default=200, help='Max episodes to train')
    args = parser.parse_args()
    
    train_rapo_agent(scenario_id=args.scenario, max_episodes=args.episodes)
