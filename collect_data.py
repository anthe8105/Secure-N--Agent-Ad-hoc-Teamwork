import time
import os
import argparse
import pandas as pd
from src.envs.LevelForagingEnv import load_default_scenario
from src.log import EstimationLogFile

def run_simulation(scenario_id, method='mcts', estimation_method='aga', max_episodes=50, resume_csv=None):
    # Load environment using the predefined scenario
    env, actual_scenario_id = load_default_scenario(method=method, scenario_id=scenario_id, display=False)
    # The ad-hoc agent uses MCTS to collect data. We can also use random if MCTS crashes, but let's stick to mcts for now.
    
    components = env.components
    
    state = env.reset()
    
    estimation_kwargs = {
        'template_types': components.get('template_types', ['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'adversary', 'deceptive_impostor']),
        'parameters_minmax': [(0.5,1.0), (0.5,1.0), (0.5,1.0)],
        'adversary_last_action': None
    }
    
    start_episode = 0
    append_mode = False
    
    if resume_csv and os.path.exists(resume_csv):
        filename = os.path.basename(resume_csv)
        try:
            exp_num = int(filename.split('_')[-1].replace('.csv', ''))
            
            df = pd.read_csv(resume_csv, sep=';', skipinitialspace=True)
            if 'Iteration' in df.columns and len(df) > 0:
                start_episode = int(df['Iteration'].dropna().max()) + 1
            else:
                start_episode = len(df)
            
            append_mode = True
            print(f"Resuming from CSV {filename}. Automatically starting at step {start_episode}.")
            max_episodes += start_episode
            
        except Exception as e:
            print(f"Error parsing resume CSV: {e}")
            exp_num = int(time.time())
    else:
        exp_num = int(time.time())
        
    log = EstimationLogFile('LevelForagingEnv', f"DataEnv_S{scenario_id}", method, estimation_method, exp_num,
                            estimation_kwargs['template_types'], estimation_kwargs['parameters_minmax'], append=append_mode)
    
    env.episode = start_episode
    print(f"Starting simulation for Scenario={scenario_id} (Target end step: {max_episodes})...")
    while env.episode < max_episodes:
        adhoc_agent = env.get_adhoc_agent()
        method_func = env.import_method(adhoc_agent.type)
        
        adhoc_agent.smart_parameters['estimation_method'] = estimation_method
        adhoc_agent.smart_parameters['estimation_kwargs'] = estimation_kwargs
        
        action, target = method_func(env, adhoc_agent)
        if type(action) is tuple and len(action) == 3: # Handle MCTS returning (action, tree, info)
            action = action[0]
        state, reward_env, done_env, info = env.step(action)
        reward = info.get('action reward', 0.0)
        done = env.state_set.is_final_state(state)
        
        typeest, parametersest = None, None
        if 'estimation' in adhoc_agent.smart_parameters:
             typeest, parametersest, _ = adhoc_agent.smart_parameters['estimation'].get_estimation(env)
             
        data = {
            'it': env.episode,
            'reward': reward,
            'time': 0.1,
            'nrollout': 0,
            'nsimulation': 0,
            'typeestimation': typeest,
            'parametersestimation': parametersest,
            'memoryusage': 0,
        }
        log.write(data)
        
        if done:
            env.respawn_tasks()
            print(f"Episode {env.episode} / {max_episodes} complete (Scenario={scenario_id})")
    
    env.close()
    print(f"Finished simulation for Scenario={scenario_id}")

if __name__ == "__main__":
    import traceback
    
    parser = argparse.ArgumentParser(description="Collect Data for Adhoc Simulation")
    parser.add_argument('--scenario', type=int, default=10, help='Scenario ID to run')
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes (steps) to execute')
    parser.add_argument('--resume', type=str, default=None, help='Path to an existing CSV file to append data to')
    args = parser.parse_args()
    
    try:
        run_simulation(scenario_id=args.scenario, max_episodes=args.episodes, resume_csv=args.resume)
    except Exception as e:
        traceback.print_exc()
