import argparse
import itertools
from collections import defaultdict
import numpy as np
import torch
from env.gridworld import GridWorld
from agents.dqn_base import DQNAgent
from agents.hierarchical_dqn import HierarchicalDQN
from agents.episodic_memory import EpisodicMemory, Episode
from explainability.trajectory_attribution import explain_action


def flatten_obs(obs: dict) -> np.ndarray:
    return obs["agent"].astype(np.float32)


# Global state visit counts for curiosity bonus
state_visit_counts: dict = {}


def get_curiosity_bonus(state: np.ndarray, beta: float = 0.5) -> float:
    """Curiosity bonus: higher for less visited states."""
    key = (int(state[0]), int(state[1]))
    count = state_visit_counts.get(key, 0)
    state_visit_counts[key] = count + 1
    return beta / np.sqrt(count + 1)


def run_episode(env: GridWorld, agent, memory: EpisodicMemory, eps: float, hierarchical: bool = False, return_then_explore: bool = False, use_curiosity: bool = True):
    obs = env.reset()
    state = flatten_obs(obs)
    done = False
    trajectory = []
    td_errors = []
    subgoals = []
    total_r = 0.0
    subgoal = None
    episode_info = {"got_key": False, "door_opened": False, "goal_reached": False}
    while not done:
        # Smart exploration: if at door with key, try use_key action with high probability
        at_door_with_key = (int(state[0]), int(state[1])) == env.door and state[2] > 0 and not env.door_opened
        if at_door_with_key and np.random.rand() < 0.5:
            action = 4  # use_key
        elif hierarchical:
            if subgoal is None or np.random.rand() < 0.1:
                subgoal = agent.select_subgoal(state, eps)
                subgoals.append(subgoal)
            action = agent.select_action(state, subgoal, eps)
        else:
            action = agent.select_action(state, eps)
        next_obs, reward, done, info = env.step(action)
        # Track events
        if info.get("got_key"):
            episode_info["got_key"] = True
        if info.get("door_opened"):
            episode_info["door_opened"] = True
        if info.get("goal_reached"):
            episode_info["goal_reached"] = True
        next_state = flatten_obs(next_obs)
        bonus = 0.0
        if hierarchical and subgoal is not None:
            dist = np.linalg.norm(next_state[:2] - np.argmax(subgoal))
            bonus = -dist * 0.1
        # Add curiosity bonus for exploration
        if use_curiosity:
            bonus += get_curiosity_bonus(next_state, beta=1.0)
        total_reward = reward + bonus
        if hierarchical:
            agent.push_low(state, subgoal, action, total_reward, next_state, done)
        else:
            agent.push(state, action, total_reward, next_state, done)
        trajectory.append((state, action, total_reward))
        td_errors.append(abs(total_reward))
        state = next_state
        total_r += total_reward
    ep = Episode(trajectory, total_r, td_errors, subgoals, timestamp=len(memory.episodes))
    if memory.should_store(ep.return_total, max(td_errors, default=0.0), sum(r > 0.5 for r in ep.rarity_scores), [e.return_total for e in memory.episodes]):
        memory.add(ep)
    return total_r, episode_info


def train(config):
    env = GridWorld(partial_observability=config.partial)
    obs_dim = 3
    action_dim = env.action_space
    memory = EpisodicMemory()
    metrics = defaultdict(list)
    successes = 0
    keys_collected = 0
    doors_opened = 0

    if config.variant in {"vanilla", "per", "memory"}:
        agent = DQNAgent(obs_dim, action_dim, prioritized=config.variant == "per")
    else:
        agent = HierarchicalDQN(obs_dim, action_dim, subgoal_dim=8)
    eps_start, eps_end, eps_decay = 1.0, 0.05, config.episodes * 0.8
    steps = 0
    for episode in range(config.episodes):
        eps = max(eps_end, eps_start - episode / eps_decay)
        use_curiosity = config.variant in {"memory", "full", "full_explain"}
        total_r, ep_info = run_episode(env, agent, memory, eps, hierarchical=config.variant in {"hier", "full", "full_explain"}, return_then_explore=config.variant in {"full", "full_explain"}, use_curiosity=use_curiosity)
        metrics["return"].append(total_r)
        if ep_info["got_key"]:
            keys_collected += 1
        if ep_info["door_opened"]:
            doors_opened += 1
        if ep_info["goal_reached"]:
            successes += 1
        if isinstance(agent, HierarchicalDQN):
            agent.update_low()
            agent.update_high()
        else:
            agent.update()
        if (episode + 1) % config.log_interval == 0:
            print(f"Episode {episode+1}: return={total_r:.1f}, eps={eps:.2f}, mem={len(memory.episodes)} | keys={keys_collected}, doors={doors_opened}, goals={successes}")
            if config.variant == "full_explain":
                obs = env.reset()
                s = flatten_obs(obs)
                act = agent.select_action(s, np.zeros(8, dtype=np.float32), eps=0.0) if isinstance(agent, HierarchicalDQN) else agent.select_action(s, eps=0.0)
                explanation = explain_action(s, act, memory)
                print("Explication:", explanation)
    if config.save:
        torch.save(metrics, config.save)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["vanilla", "per", "memory", "hier", "full", "full_explain"], default="vanilla")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
