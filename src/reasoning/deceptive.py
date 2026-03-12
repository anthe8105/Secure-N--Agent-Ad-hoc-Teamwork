import random as rd
import numpy as np
from src.utils.math import euclidean_distance

def deceptive_planning(env, agent):
    """
    Rule-based deceptive adversary:
    - Early phase: Cooperate (move towards nearest task and load).
    - Late phase: Sabotage (move towards the ad-hoc agent to block/hover).
    """
    tasks = env.components['tasks']
    uncompleted_tasks = [t for t in tasks if not t.completed]
    
    total_tasks = len(tasks)
    # Sabotage starts when <= 30% of tasks remain (at least <= 2)
    is_late_phase = (len(uncompleted_tasks) <= max(2, int(total_tasks * 0.3)))

    adhoc_agent = None
    if 'adhoc_agent_index' in env.components:
        for a in env.components['agents']:
            if a.index == env.components['adhoc_agent_index']:
                adhoc_agent = a
                break

    if is_late_phase and adhoc_agent is not None:
        # SABOTAGE / HOVERING
        target_pos = adhoc_agent.position
        best_action = 4 # Load (does nothing if no task)
        min_dist = float('inf')
        for a in [0, 1, 2, 3]:
            if a == 0: next_pos = (agent.position[0]+1, agent.position[1])
            elif a == 1: next_pos = (agent.position[0]-1, agent.position[1])
            elif a == 2: next_pos = (agent.position[0], agent.position[1]+1)
            elif a == 3: next_pos = (agent.position[0], agent.position[1]-1)
            
            if 0 <= next_pos[0] < env.shape[0] and 0 <= next_pos[1] < env.shape[1]:
                d = euclidean_distance(next_pos, target_pos)
                if d < min_dist:
                    min_dist = d
                    best_action = a
        return best_action, None
    else:
        # COOPERATE
        if not uncompleted_tasks:
            return rd.choice([0,1,2,3]), None
            
        target_task = min(uncompleted_tasks, key=lambda t: euclidean_distance(agent.position, t.position))
        target_pos = target_task.position
        
        if target_pos == agent.position:
            return 4, target_task.position # Load
            
        min_dist = float('inf')
        best_action = rd.choice([0,1,2,3])
        for a in [0, 1, 2, 3]:
            if a == 0: next_pos = (agent.position[0]+1, agent.position[1])
            elif a == 1: next_pos = (agent.position[0]-1, agent.position[1])
            elif a == 2: next_pos = (agent.position[0], agent.position[1]+1)
            elif a == 3: next_pos = (agent.position[0], agent.position[1]-1)
            
            if 0 <= next_pos[0] < env.shape[0] and 0 <= next_pos[1] < env.shape[1]:
                d = euclidean_distance(next_pos, target_pos)
                if d < min_dist:
                    min_dist = d
                    best_action = a
        return best_action, target_task.position

def deceptive_impostor_planning(env, agent):
    # wrapper for naming convention (main_type: deceptive)
    return deceptive_planning(env, agent)
    
def deceptive_adversary_planning(env, agent):
    return deceptive_planning(env, agent)
