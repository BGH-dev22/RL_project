"""
Compare all 6 variants and generate analysis plots.
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.gridworld import GridWorld
from agents.dqn_base import DQNAgent
from agents.hierarchical_dqn import HierarchicalDQN
from agents.episodic_memory import EpisodicMemory, Episode
from explainability.trajectory_attribution import explain_action


# Global state visit counts for curiosity bonus
state_visit_counts: dict = {}


def reset_curiosity():
    global state_visit_counts
    state_visit_counts = {}


def get_curiosity_bonus(state: np.ndarray, beta: float = 0.5) -> float:
    key = (int(state[0]), int(state[1]))
    count = state_visit_counts.get(key, 0)
    state_visit_counts[key] = count + 1
    return beta / np.sqrt(count + 1)


def flatten_obs(obs: dict) -> np.ndarray:
    return obs["agent"].astype(np.float32)


# Subgoal targets for hierarchical agent
SUBGOAL_TARGETS = {
    0: (1, 8),   # Key position
    1: (5, 5),   # Door position  
    2: (8, 8),   # Goal position
    3: (2, 2),   # Corner 1
    4: (2, 7),   # Corner 2
    5: (7, 2),   # Corner 3
    6: (7, 7),   # Corner 4
    7: (5, 2),   # Alternative door area
}


def run_episode(env: GridWorld, agent, memory: EpisodicMemory, eps: float, 
                hierarchical: bool = False, use_curiosity: bool = True,
                episode_num: int = 0, total_episodes: int = 3000):
    obs = env.reset()
    state = flatten_obs(obs)
    done = False
    trajectory = []
    td_errors = []
    subgoals = []
    total_r = 0.0
    subgoal = None
    subgoal_idx = None
    subgoal_steps = 0
    subgoal_start_state = None
    episode_info = {"got_key": False, "door_opened": False, "goal_reached": False, "steps": 0}
    
    # Progressive hierarchy: start simple, add complexity gradually
    hierarchy_ratio = min(1.0, episode_num / (total_episodes * 0.3))  # Full hierarchy after 30% episodes
    use_hierarchy_now = hierarchical and (np.random.rand() < hierarchy_ratio)
    
    while not done:
        episode_info["steps"] += 1
        
        # Smart exploration for door - higher probability
        at_door_with_key = (int(state[0]), int(state[1])) == env.door and state[2] > 0 and not env.door_opened
        if at_door_with_key and np.random.rand() < 0.8:  # Increased from 0.5
            action = 4
        elif use_hierarchy_now:
            # Subgoal selection with temporal abstraction
            if subgoal is None or subgoal_steps >= 15 or np.random.rand() < 0.05:
                # Context-aware subgoal: prioritize useful subgoals
                if state[2] == 0 and not episode_info["got_key"]:
                    # Don't have key -> go to key
                    subgoal_idx = 0
                elif state[2] > 0 and not env.door_opened:
                    # Have key, door closed -> go to door
                    subgoal_idx = 1
                elif env.door_opened:
                    # Door open -> go to goal
                    subgoal_idx = 2
                else:
                    subgoal_idx = agent.select_subgoal(state, eps * 0.5).argmax()
                
                subgoal = np.zeros(agent.subgoal_dim, dtype=np.float32)
                subgoal[subgoal_idx] = 1.0
                subgoals.append(subgoal)
                subgoal_steps = 0
                subgoal_start_state = state.copy()
            
            action = agent.select_action(state, subgoal, eps)
            subgoal_steps += 1
        elif hierarchical:
            # Fallback to simple DQN-like behavior using low-level with default subgoal
            if subgoal is None:
                subgoal = np.zeros(agent.subgoal_dim, dtype=np.float32)
                subgoal[0] = 1.0  # Default subgoal
            action = agent.select_action(state, subgoal, eps)
        else:
            action = agent.select_action(state, eps)
        
        next_obs, reward, done, info = env.step(action)
        if info.get("got_key"):
            episode_info["got_key"] = True
        if info.get("door_opened"):
            episode_info["door_opened"] = True
        if info.get("goal_reached"):
            episode_info["goal_reached"] = True
        
        next_state = flatten_obs(next_obs)
        
        # Improved bonus calculation
        bonus = 0.0
        
        # Subgoal-based intrinsic reward (only when using hierarchy)
        if use_hierarchy_now and subgoal is not None and subgoal_idx is not None:
            target = SUBGOAL_TARGETS.get(subgoal_idx, (5, 5))
            old_dist = np.sqrt((state[0] - target[0])**2 + (state[1] - target[1])**2)
            new_dist = np.sqrt((next_state[0] - target[0])**2 + (next_state[1] - target[1])**2)
            # Reward for getting closer to subgoal
            bonus += (old_dist - new_dist) * 0.5
            # Bonus for reaching subgoal
            if new_dist < 1.0:
                bonus += 2.0
        
        # Curiosity bonus
        if use_curiosity:
            bonus += get_curiosity_bonus(next_state, beta=0.5)  # Reduced from 1.0
        
        total_reward = reward + bonus
        
        if hierarchical:
            agent.push_low(state, subgoal, action, total_reward, next_state, done)
            # Train high-level less frequently
            if done or subgoal_steps >= 15:
                if subgoal_start_state is not None:
                    high_reward = total_r if done else bonus * 5
                    agent.push_high(subgoal_start_state, subgoal, high_reward, next_state, done)
        else:
            agent.push(state, action, total_reward, next_state, done)
        
        trajectory.append((state, action, total_reward))
        td_errors.append(abs(total_reward))
        state = next_state
        total_r += total_reward
    
    ep = Episode(trajectory, total_r, td_errors, subgoals, timestamp=len(memory.episodes))
    if memory.should_store(ep.return_total, max(td_errors, default=0.0), 
                           sum(r > 0.5 for r in ep.rarity_scores), 
                           [e.return_total for e in memory.episodes]):
        memory.add(ep)
    
    return total_r, episode_info


def train_variant(variant: str, episodes: int, seed: int = 42) -> dict:
    """Train a single variant and return metrics."""
    print(f"\n{'='*50}")
    print(f"Training variant: {variant} ({episodes} episodes)")
    print(f"{'='*50}")
    
    np.random.seed(seed)
    reset_curiosity()
    
    env = GridWorld(seed=seed)
    obs_dim = 3
    action_dim = env.action_space
    memory = EpisodicMemory()
    
    metrics = {
        "returns": [],
        "keys": [],
        "doors": [],
        "goals": [],
        "steps": [],
        "cumulative_goals": [],
    }
    
    hierarchical = variant in {"hier", "full", "full_explain"}
    use_curiosity = variant in {"memory", "full", "full_explain"}
    prioritized = variant in {"per", "memory", "full", "full_explain"}
    
    if variant in {"vanilla", "per", "memory"}:
        agent = DQNAgent(obs_dim, action_dim, prioritized=prioritized)
    else:
        agent = HierarchicalDQN(obs_dim, action_dim, subgoal_dim=8)
    
    # Slower epsilon decay for hierarchical variants (more exploration needed)
    if hierarchical:
        eps_start, eps_end, eps_decay = 1.0, 0.1, episodes * 0.6  # Higher final eps, faster decay
    else:
        eps_start, eps_end, eps_decay = 1.0, 0.05, episodes * 0.8
    cumulative_goals = 0
    
    for episode in range(episodes):
        eps = max(eps_end, eps_start - episode / eps_decay)
        total_r, ep_info = run_episode(env, agent, memory, eps, 
                                        hierarchical=hierarchical, 
                                        use_curiosity=use_curiosity,
                                        episode_num=episode,
                                        total_episodes=episodes)
        
        metrics["returns"].append(total_r)
        metrics["keys"].append(1 if ep_info["got_key"] else 0)
        metrics["doors"].append(1 if ep_info["door_opened"] else 0)
        metrics["goals"].append(1 if ep_info["goal_reached"] else 0)
        metrics["steps"].append(ep_info["steps"])
        
        if ep_info["goal_reached"]:
            cumulative_goals += 1
        metrics["cumulative_goals"].append(cumulative_goals)
        
        if isinstance(agent, HierarchicalDQN):
            agent.update_low()
            agent.update_high()
        else:
            agent.update()
        
        if (episode + 1) % (episodes // 10) == 0:
            success_rate = cumulative_goals / (episode + 1) * 100
            print(f"  Episode {episode+1}: return={total_r:.1f}, goals={cumulative_goals} ({success_rate:.1f}%)")
    
    return metrics


def smooth(data, window=50):
    """Smooth data with moving average."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_comparison(all_metrics: dict, save_dir: str):
    """Generate comparison plots."""
    os.makedirs(save_dir, exist_ok=True)
    
    variants = list(all_metrics.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(variants)))
    
    # 1. Returns over episodes
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (variant, metrics) in enumerate(all_metrics.items()):
        smoothed = smooth(metrics["returns"], window=50)
        ax.plot(smoothed, label=variant, color=colors[i], linewidth=2)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Return (smoothed)", fontsize=12)
    ax.set_title("Courbe de Retour par Variante", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "returns_comparison.png"), dpi=150)
    plt.close()
    
    # 2. Success rate over time
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (variant, metrics) in enumerate(all_metrics.items()):
        episodes = np.arange(1, len(metrics["cumulative_goals"]) + 1)
        success_rate = np.array(metrics["cumulative_goals"]) / episodes * 100
        ax.plot(success_rate, label=variant, color=colors[i], linewidth=2)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Taux de Succès Cumulé (%)", fontsize=12)
    ax.set_title("Taux de Succès par Variante", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "success_rate.png"), dpi=150)
    plt.close()
    
    # 3. Bar chart: Final metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    final_success = [sum(m["goals"]) / len(m["goals"]) * 100 for m in all_metrics.values()]
    final_doors = [sum(m["doors"]) / len(m["doors"]) * 100 for m in all_metrics.values()]
    final_keys = [sum(m["keys"]) / len(m["keys"]) * 100 for m in all_metrics.values()]
    
    x = np.arange(len(variants))
    width = 0.6
    
    axes[0].bar(x, final_keys, width, color=colors)
    axes[0].set_ylabel("Taux (%)")
    axes[0].set_title("Clés Collectées")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(variants, rotation=45, ha="right")
    
    axes[1].bar(x, final_doors, width, color=colors)
    axes[1].set_ylabel("Taux (%)")
    axes[1].set_title("Portes Ouvertes")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(variants, rotation=45, ha="right")
    
    axes[2].bar(x, final_success, width, color=colors)
    axes[2].set_ylabel("Taux (%)")
    axes[2].set_title("Goals Atteints")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(variants, rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "final_metrics.png"), dpi=150)
    plt.close()
    
    # 4. Sample efficiency
    fig, ax = plt.subplots(figsize=(10, 6))
    threshold = 0.5  # 50% success rate
    episodes_to_threshold = []
    for variant, metrics in all_metrics.items():
        episodes = np.arange(1, len(metrics["cumulative_goals"]) + 1)
        success_rate = np.array(metrics["cumulative_goals"]) / episodes
        reached = np.where(success_rate >= threshold)[0]
        if len(reached) > 0:
            episodes_to_threshold.append(reached[0] + 1)
        else:
            episodes_to_threshold.append(len(metrics["goals"]))
    
    ax.bar(x, episodes_to_threshold, width, color=colors)
    ax.set_ylabel("Épisodes")
    ax.set_title(f"Sample Efficiency: Épisodes pour atteindre {int(threshold*100)}% de succès")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=45, ha="right")
    ax.axhline(y=len(list(all_metrics.values())[0]["goals"]), color='r', linestyle='--', label="Max episodes")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sample_efficiency.png"), dpi=150)
    plt.close()
    
    # 5. Return distribution (box plot)
    fig, ax = plt.subplots(figsize=(12, 6))
    data = [metrics["returns"][-500:] for metrics in all_metrics.values()]  # Last 500 episodes
    bp = ax.boxplot(data, labels=variants, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("Return")
    ax.set_title("Distribution des Retours (500 derniers épisodes)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "return_distribution.png"), dpi=150)
    plt.close()
    
    print(f"\nGraphiques sauvegardés dans: {save_dir}")


def generate_summary_table(all_metrics: dict, save_path: str):
    """Generate summary statistics table."""
    summary = []
    for variant, metrics in all_metrics.items():
        n = len(metrics["goals"])
        total_goals = sum(metrics["goals"])
        total_doors = sum(metrics["doors"])
        total_keys = sum(metrics["keys"])
        returns = metrics["returns"]
        
        summary.append({
            "Variante": variant,
            "Episodes": n,
            "Clés (%)": f"{total_keys/n*100:.1f}",
            "Portes (%)": f"{total_doors/n*100:.1f}",
            "Goals (%)": f"{total_goals/n*100:.1f}",
            "Retour Moyen": f"{np.mean(returns):.1f}",
            "Retour Std": f"{np.std(returns):.1f}",
            "Retour Max": f"{max(returns):.1f}",
        })
    
    # Print table
    print("\n" + "="*100)
    print("RÉSUMÉ DES PERFORMANCES")
    print("="*100)
    headers = list(summary[0].keys())
    print(" | ".join(f"{h:>12}" for h in headers))
    print("-"*100)
    for row in summary:
        print(" | ".join(f"{v:>12}" for v in row.values()))
    print("="*100)
    
    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Résumé sauvegardé: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare all 6 DQN variants")
    parser.add_argument("--episodes", type=int, default=3000, help="Episodes per variant")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    variants = ["vanilla", "per", "memory", "hier", "full", "full_explain"]
    all_metrics = {}
    
    for variant in variants:
        metrics = train_variant(variant, args.episodes, seed=args.seed)
        all_metrics[variant] = metrics
        
        # Save individual results
        with open(output_dir / f"{variant}_metrics.json", 'w') as f:
            json.dump({k: [float(x) for x in v] for k, v in metrics.items()}, f)
    
    # Generate plots and summary
    plot_comparison(all_metrics, str(output_dir / "plots"))
    generate_summary_table(all_metrics, str(output_dir / "summary.json"))
    
    print(f"\nTous les résultats sauvegardés dans: {output_dir}")


if __name__ == "__main__":
    main()
