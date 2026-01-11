from typing import Dict, List, Tuple
import numpy as np
from agents.episodic_memory import EpisodicMemory


def explain_action(state: np.ndarray, action: int, memory: EpisodicMemory, k: int = 5) -> str:
    candidates = memory.similar_episodes(state, k)
    if not candidates:
        return "Aucune trajectoire mémorisée pertinente."
    # Count how many trajectories (not transitions) take each action at similar states
    traj_actions: Dict[int, List[int]] = {}  # action -> list of trajectory indices
    returns: Dict[int, List[float]] = {}
    for idx, ep in enumerate(candidates):
        for (s, a, r) in ep.trajectory:
            if np.linalg.norm(s[:2] - state[:2]) < 1.5:
                if a not in traj_actions:
                    traj_actions[a] = []
                if idx not in traj_actions[a]:
                    traj_actions[a].append(idx)
                    returns.setdefault(a, []).append(ep.return_total)
                break  # count each trajectory only once per action
    if not traj_actions:
        return "Trajectoires similaires trouvées mais sans décisions alignées."
    best_action = max(traj_actions.items(), key=lambda kv: len(kv[1]))[0]
    freq = len(traj_actions.get(action, []))
    ret_mean = np.mean(returns.get(action, [0.0])) if action in returns else 0.0
    return f"Action {action} choisie car {freq}/{len(candidates)} trajectoires similaires la prennent (retour moyen {ret_mean:.1f}). Meilleure action mémorisée: {best_action}."
