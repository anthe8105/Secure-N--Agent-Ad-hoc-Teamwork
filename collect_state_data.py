"""
collect_state_data.py
---------------------
Data collection script for state-based adversary detection (DBBC alignment).

Instead of logging ad-hoc estimation outputs, this script logs raw (observation,
action) pairs per uncontrolled agent at each environment step, matching the paper's
local history h_t^i = (o_0, a_0, ..., o_t) fed into the recurrent encoder Enc_phi.

Observation vector (obs_dim=11) per uncontrolled agent j observed by adhoc agent A:
    [j_x, j_y,                        # absolute position of j (2)
     A_x - j_x, A_y - j_y,            # relative position to adhoc (2)
     T0_x - j_x, T0_y - j_y,          # relative position to task 0 (2)
     T1_x - j_x, T1_y - j_y,          # relative position to task 1 (2)
     float(task0.completed),           # task completion flags (2)
     float(task1.completed)]
Action: integer 0-4, stored as-is (converted to 1-hot in the Dataset)
Label: IsAdversary = 1 if agent.index == impostor_index else 0

Output CSV columns: Step, AgentIndex, IsAdversary, StateVec, Action

Designed for Scenario 13 (3x3, 3 agents, 2 tasks) but works for any scenario
where there are exactly 2 tasks.
"""
import os
import time
import argparse
import pandas as pd
from src.envs.LevelForagingEnv import load_default_scenario

OBS_DIM = 10   # Fixed observation vector length (see docstring)
N_TASKS = 2    # Number of tasks assumed in the scenario


def build_obs_vector(agent, adhoc_agent, tasks):
    """
    Build a fixed-dim observation vector for uncontrolled agent 'agent'
    from the perspective of 'adhoc_agent'.

    Returns a list of OBS_DIM floats.
    """
    j_x, j_y = agent.position
    a_x, a_y = adhoc_agent.position

    # Pad tasks list to N_TASKS entries (in case some are already completed/removed)
    padded_tasks = list(tasks) + [None] * N_TASKS
    padded_tasks = padded_tasks[:N_TASKS]

    obs = [float(j_x), float(j_y),
           float(a_x - j_x), float(a_y - j_y)]

    for task in padded_tasks:
        if task is not None:
            t_x, t_y = task.position
            obs.extend([float(t_x - j_x), float(t_y - j_y)])
        else:
            obs.extend([0.0, 0.0])

    for task in padded_tasks:
        obs.append(float(task.completed) if task is not None else 1.0)

    assert len(obs) == OBS_DIM, f"Expected obs length {OBS_DIM}, got {len(obs)}"
    return obs


def run_simulation(scenario_id, max_steps=500, resume_csv=None):
    env, actual_id = load_default_scenario(method='mcts', scenario_id=scenario_id, display=False)

    if not env.is_adversarial():
        raise ValueError(f"Scenario {scenario_id} has no impostor_index set. "
                         "Please use an adversarial scenario (e.g. 13).")

    # ---- Output CSV setup ----
    os.makedirs('results', exist_ok=True)
    exp_num = int(time.time())
    if resume_csv and os.path.exists(resume_csv):
        out_path = resume_csv
        append_mode = True
        existing = pd.read_csv(out_path, sep=';')
        start_step = int(existing['Step'].max()) + 1 if 'Step' in existing.columns and len(existing) > 0 else 0
        print(f"Resuming from {out_path}, starting at step {start_step}")
    else:
        out_path = f"results/state_data_{actual_id}_{exp_num}.csv"
        append_mode = False
        start_step = 0

    if not append_mode:
        with open(out_path, 'w') as f:
            f.write("Step;AgentIndex;IsAdversary;StateVec;Action\n")

    print(f"Collecting state-action data for Scenario {actual_id} → {out_path}")
    print(f"obs_dim={OBS_DIM}, action_dim=5 (1-hot), max_steps={max_steps}")

    # ---- Simulation loop ----
    env.reset()
    impostor_index = env.components['impostor_index']
    step = start_step

    while step < start_step + max_steps:
        adhoc_agent = env.get_adhoc_agent()
        method_func = env.import_method(adhoc_agent.type)

        # Compute adhoc action
        action_result = method_func(env, adhoc_agent)
        if isinstance(action_result, tuple) and len(action_result) >= 1:
            adhoc_action = action_result[0]
            if isinstance(adhoc_action, tuple):  # MCTS may return (action, tree, info)
                adhoc_action = adhoc_action[0]
        else:
            adhoc_action = action_result

        # Before stepping: record (obs, action) for each uncontrolled agent
        # (Their next_action was set during the previous transition or is None at t=0)
        all_tasks = env.components['tasks']
        rows = []

        for agent in env.components['agents']:
            if agent.index == adhoc_agent.index:
                continue  # Skip the strategic adhoc agent itself

            is_adv = 1 if agent.index == impostor_index else 0

            obs_vec = build_obs_vector(agent, adhoc_agent, all_tasks)

            # next_action is None on the very first step; default to 4 (Load / no-op)
            agent_action = agent.next_action if agent.next_action is not None else 4

            rows.append({
                'Step': step,
                'AgentIndex': agent.index,
                'IsAdversary': is_adv,
                'StateVec': str(obs_vec),
                'Action': int(agent_action),
            })

        # Write rows to CSV
        if rows:
            with open(out_path, 'a') as f:
                for row in rows:
                    f.write(f"{row['Step']};{row['AgentIndex']};{row['IsAdversary']};"
                            f"{row['StateVec']};{row['Action']}\n")

        # Step the environment
        state, reward, done, info = env.step(adhoc_action)
        step += 1

        if done or env.state_set.is_final_state(state):
            env.respawn_tasks()
            all_tasks = env.components['tasks']

    env.close()
    print(f"\nDone. Collected {step - start_step} steps → {out_path}")
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect raw (state, action) data for adversary detection")
    parser.add_argument('--scenario', type=int, default=13,
                        help='Scenario ID (default: 13 = 3x3 minimal adversarial)')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of environment steps to collect')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to existing state_data CSV to append to')
    args = parser.parse_args()

    import traceback
    try:
        run_simulation(scenario_id=args.scenario, max_steps=args.episodes, resume_csv=args.resume)
    except Exception as e:
        traceback.print_exc()
