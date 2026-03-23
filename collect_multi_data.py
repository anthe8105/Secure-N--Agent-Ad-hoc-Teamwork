"""
collect_multi_data.py
---------------------
Data collection for multi-observer adversary detection (Scenario 14+).

For each env step, logs one row per (observer_i, target_j) pair:
  - observer_i ∈ strategic agents  (e.g. A1, A2)
  - target_j   ∈ uncontrolled agents (e.g. T, X1, X2)

CSV columns:
  Step; ObserverId; AgentIndex; IsAdversary; StateVec; Action; RelFeatures

StateVec per pair (i,j), obs_dim varies by N_TASKS:
  [j_x, j_y,
   i_x - j_x, i_y - j_y,               # relative to observer
   τ0_x - j_x, τ0_y - j_y,             # relative to each task
   τ1_x - j_x, τ1_y - j_y,
   ...
   done_0, done_1, ...]                  # task completion flags

RelFeatures (rel_feat_dim = 3):
  [i_x - j_x, i_y - j_y, ||i - j||_2]

Action: integer 0-4 (converted to 1-hot in dataset)

NOTE: This script preserves the existing collect_state_data.py for Scenario 13.
"""
import os
import math
import time
import argparse
import pandas as pd
from src.envs.LevelForagingEnv import load_default_scenario


# ------------- Configurable per scenario -----------------
# Scenario 14: 5x5, 3 tasks → obs_dim = 2 + 2 + 2*3 + 3 = 13
# but these are auto-detected from the env at runtime.
ACTION_DIM = 5


def build_obs_vector(agent_j, observer_i, tasks, n_tasks):
    """
    Build a fixed-dim observation vector for uncontrolled agent `agent_j`
    observed by strategic agent `observer_i`.

    Returns (obs_vec, rel_features).
    """
    j_x, j_y = agent_j.position
    i_x, i_y = observer_i.position

    obs = [float(j_x), float(j_y),
           float(i_x - j_x), float(i_y - j_y)]

    # Pad / truncate task list to exactly n_tasks
    padded_tasks = list(tasks) + [None] * n_tasks
    padded_tasks = padded_tasks[:n_tasks]

    for task in padded_tasks:
        if task is not None:
            t_x, t_y = task.position
            obs.extend([float(t_x - j_x), float(t_y - j_y)])
        else:
            obs.extend([0.0, 0.0])

    for task in padded_tasks:
        obs.append(float(task.completed) if task is not None else 1.0)

    # Relational features ψ^{ij}_t
    dx = float(i_x - j_x)
    dy = float(i_y - j_y)
    dist = math.sqrt(dx * dx + dy * dy)
    rel_features = [dx, dy, dist]

    return obs, rel_features


def detect_strategic_agents(env):
    """
    Returns the list of strategic agent indices.
    Heuristic: the adhoc agent + any agent with 'l4' type (cooperative observer).
    Override for custom setups.
    """
    adhoc_idx = env.components['adhoc_agent_index']
    strategic = [adhoc_idx]
    for agent in env.components['agents']:
        if agent.index != adhoc_idx and agent.type == 'l4':
            strategic.append(agent.index)
    return strategic


def run_simulation(scenario_id, max_steps=500, strategic_indices=None, method='mcts'):
    env, actual_id = load_default_scenario(method=method, scenario_id=scenario_id, display=False)

    if not env.is_adversarial():
        raise ValueError(f"Scenario {scenario_id} has no impostor_index set.")

    # Auto-detect strategic agents if not specified
    if strategic_indices is None:
        strategic_indices = detect_strategic_agents(env)
    print(f"Strategic agents: {strategic_indices}")

    # Determine number of tasks
    n_tasks = len(env.components['tasks'])
    obs_dim = 2 + 2 + 2 * n_tasks + n_tasks   # pos + rel_to_observer + rel_to_tasks + flags
    print(f"n_tasks={n_tasks}, obs_dim={obs_dim}, action_dim={ACTION_DIM}, rel_feat_dim=3")

    # Output CSV
    os.makedirs('results', exist_ok=True)
    exp_num = int(time.time())
    out_path = f"results/multi_state_data_{actual_id}_{exp_num}.csv"
    with open(out_path, 'w') as f:
        f.write("Step;ObserverId;AgentIndex;IsAdversary;StateVec;Action;RelFeatures\n")

    print(f"Collecting multi-observer data → {out_path}")
    print(f"max_steps={max_steps}")

    # Determine impostor indices (string or list)
    imp_idx = env.components['impostor_index']

    # Simulation loop
    env.reset()
    step = 0

    while step < max_steps:
        adhoc_agent = env.get_adhoc_agent()
        method_func = env.import_method(adhoc_agent.type)

        # Compute adhoc action
        action_result = method_func(env, adhoc_agent)
        if isinstance(action_result, tuple) and len(action_result) >= 1:
            adhoc_action = action_result[0]
            if isinstance(adhoc_action, tuple):
                adhoc_action = adhoc_action[0]
        else:
            adhoc_action = action_result

        # Identify uncontrolled agents (everyone not in strategic set)
        all_tasks = env.components['tasks']
        rows = []

        for observer_idx in strategic_indices:
            observer_agent = None
            for a in env.components['agents']:
                if a.index == observer_idx:
                    observer_agent = a
                    break
            if observer_agent is None:
                continue

            for agent in env.components['agents']:
                if agent.index in strategic_indices:
                    continue  # Skip strategic agents — they are observers, not targets

                # Is this agent an impostor?
                if isinstance(imp_idx, list):
                    is_adv = 1 if agent.index in imp_idx else 0
                else:
                    is_adv = 1 if agent.index == imp_idx else 0

                obs_vec, rel_feat = build_obs_vector(agent, observer_agent, all_tasks, n_tasks)

                agent_action = agent.next_action if agent.next_action is not None else 4

                rows.append({
                    'Step': step,
                    'ObserverId': observer_idx,
                    'AgentIndex': agent.index,
                    'IsAdversary': is_adv,
                    'StateVec': str(obs_vec),
                    'Action': int(agent_action),
                    'RelFeatures': str(rel_feat),
                })

        # Write to CSV
        if rows:
            with open(out_path, 'a') as f:
                for row in rows:
                    f.write(f"{row['Step']};{row['ObserverId']};{row['AgentIndex']};"
                            f"{row['IsAdversary']};{row['StateVec']};{row['Action']};"
                            f"{row['RelFeatures']}\n")

        # Step the environment
        state, reward, done, info = env.step(adhoc_action)
        step += 1

        if done or env.state_set.is_final_state(state):
            env.respawn_tasks()

    env.close()
    print(f"\nDone. Collected {step} steps → {out_path}")
    print(f"  Rows per step: {len(strategic_indices)} observers × "
          f"{len(env.components['agents']) - len(strategic_indices)} targets = "
          f"{len(strategic_indices) * (len(env.components['agents']) - len(strategic_indices))}")
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Collect multi-observer (state, action) data for adversary detection")
    parser.add_argument('--scenario', type=int, default=14,
                        help='Scenario ID (default: 14 = 5x5 multi-agent)')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of environment steps to collect')
    parser.add_argument('--strategic', type=str, nargs='*', default=None,
                        help='List of strategic agent indices (auto-detected if omitted)')
    parser.add_argument('--method', type=str, default='mcts',
                        help="Adhoc agent's planning method (default: mcts, use l4 for mcts_min adversaries)")
    args = parser.parse_args()

    import traceback
    try:
        run_simulation(
            scenario_id=args.scenario,
            max_steps=args.episodes,
            strategic_indices=args.strategic,
            method=args.method)
    except Exception as e:
        traceback.print_exc()
