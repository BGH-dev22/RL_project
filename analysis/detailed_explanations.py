"""
Detailed analysis of trajectory-based explanations.
"""
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.gridworld import GridWorld
from agents.dqn_base import DQNAgent
from agents.hierarchical_dqn import HierarchicalDQN
from agents.episodic_memory import EpisodicMemory, Episode
from explainability.trajectory_attribution import explain_action


def flatten_obs(obs):
    return obs["agent"].astype(np.float32)


def analyze_explanations(agent, memory: EpisodicMemory, env: GridWorld, n_samples: int = 20):
    """Analyze explanation quality across different states."""
    results = []
    
    # Sample states from different regions
    test_positions = [
        (0, 0),   # Start
        (1, 8),   # Key location
        (4, 5),   # Near door
        (5, 5),   # Door
        (8, 8),   # Goal
        (2, 2),   # Trap area
        (3, 3),   # Mid-grid
    ]
    
    for pos in test_positions:
        for has_key in [False, True]:
            state = np.array([pos[0], pos[1], int(has_key)], dtype=np.float32)
            
            if isinstance(agent, HierarchicalDQN):
                subgoal = np.zeros(8, dtype=np.float32)
                subgoal[0] = 1.0
                action = agent.select_action(state, subgoal, eps=0.0)
            else:
                action = agent.select_action(state, eps=0.0)
            
            explanation = explain_action(state, action, memory, k=5)
            
            # Get similar trajectories for analysis
            similar_eps = memory.similar_episodes(state, k=5)
            avg_return = np.mean([ep.return_total for ep in similar_eps]) if similar_eps else 0
            
            results.append({
                "position": pos,
                "has_key": has_key,
                "action": action,
                "action_name": GridWorld.ACTIONS[action],
                "explanation": explanation,
                "similar_eps_count": len(similar_eps),
                "similar_eps_avg_return": avg_return,
            })
    
    return results


def generate_explanation_report(results: list, save_path: str):
    """Generate a detailed explanation analysis report."""
    report = []
    report.append("=" * 80)
    report.append("ANALYSE DES EXPLICATIONS PAR TRAJECTOIRES")
    report.append("=" * 80)
    report.append("")
    
    for i, r in enumerate(results):
        report.append(f"Cas {i+1}: Position {r['position']}, Clé: {r['has_key']}")
        report.append(f"  Action choisie: {r['action']} ({r['action_name']})")
        report.append(f"  Épisodes similaires: {r['similar_eps_count']}")
        report.append(f"  Retour moyen similaires: {r['similar_eps_avg_return']:.1f}")
        report.append(f"  Explication: {r['explanation']}")
        report.append("-" * 40)
    
    report_text = "\n".join(report)
    print(report_text)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    return report_text


def plot_explanation_heatmap(memory: EpisodicMemory, env: GridWorld, save_path: str):
    """Create heatmap of trajectory coverage."""
    grid_size = env.size
    visit_counts = np.zeros((grid_size, grid_size))
    success_counts = np.zeros((grid_size, grid_size))
    
    for ep in memory.episodes:
        for (state, action, reward) in ep.trajectory:
            y, x = int(state[0]), int(state[1])
            if 0 <= y < grid_size and 0 <= x < grid_size:
                visit_counts[y, x] += 1
                if ep.return_total > 0:
                    success_counts[y, x] += 1
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Visit frequency
    im1 = axes[0].imshow(visit_counts, cmap='Blues', origin='upper')
    axes[0].set_title('Fréquence de Visite')
    plt.colorbar(im1, ax=axes[0])
    
    # Success rate per cell
    with np.errstate(divide='ignore', invalid='ignore'):
        success_rate = np.where(visit_counts > 0, success_counts / visit_counts, 0)
    im2 = axes[1].imshow(success_rate, cmap='Greens', origin='upper', vmin=0, vmax=1)
    axes[1].set_title('Taux de Succès par Cellule')
    plt.colorbar(im2, ax=axes[1])
    
    # Environment layout
    axes[2].imshow(env.grid, cmap='coolwarm', origin='upper')
    axes[2].set_title('Layout Environnement')
    
    # Add annotations
    for ax in axes:
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Heatmap sauvegardée: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=str, default="results/full_explain_metrics.json")
    parser.add_argument("--output", type=str, default="results/explanations")
    parser.add_argument("--episodes", type=int, default=500, help="Episodes to train for analysis")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train a model for explanation analysis
    print("Training model for explanation analysis...")
    
    np.random.seed(42)
    env = GridWorld(seed=42)
    obs_dim = 3
    action_dim = env.action_space
    memory = EpisodicMemory()
    agent = HierarchicalDQN(obs_dim, action_dim, subgoal_dim=8)
    
    # Quick training loop
    state_visit_counts = {}
    for episode in range(args.episodes):
        obs = env.reset()
        state = flatten_obs(obs)
        done = False
        trajectory = []
        td_errors = []
        total_r = 0.0
        subgoal = agent.select_subgoal(state, eps=0.5)
        
        while not done:
            at_door = (int(state[0]), int(state[1])) == env.door and state[2] > 0 and not env.door_opened
            if at_door and np.random.rand() < 0.5:
                action = 4
            else:
                action = agent.select_action(state, subgoal, eps=max(0.05, 1.0 - episode/400))
            
            next_obs, reward, done, info = env.step(action)
            next_state = flatten_obs(next_obs)
            
            # Curiosity bonus
            key = (int(next_state[0]), int(next_state[1]))
            count = state_visit_counts.get(key, 0)
            state_visit_counts[key] = count + 1
            bonus = 1.0 / np.sqrt(count + 1)
            
            total_reward = reward + bonus
            agent.push_low(state, subgoal, action, total_reward, next_state, done)
            trajectory.append((state, action, total_reward))
            td_errors.append(abs(total_reward))
            state = next_state
            total_r += total_reward
        
        ep = Episode(trajectory, total_r, td_errors, [subgoal], timestamp=episode)
        if memory.should_store(ep.return_total, max(td_errors, default=0.0), 
                               sum(r > 0.5 for r in ep.rarity_scores),
                               [e.return_total for e in memory.episodes]):
            memory.add(ep)
        
        agent.update_low()
        agent.update_high()
        
        if (episode + 1) % 100 == 0:
            print(f"  Episode {episode+1}: return={total_r:.1f}, memory={len(memory.episodes)}")
    
    print(f"\nAnalyzing explanations with {len(memory.episodes)} episodes in memory...")
    
    # Analyze explanations
    results = analyze_explanations(agent, memory, env)
    generate_explanation_report(results, str(output_dir / "explanation_report.txt"))
    
    # Generate heatmap
    plot_explanation_heatmap(memory, env, str(output_dir / "trajectory_heatmap.png"))
    
    # Save results as JSON
    json_results = [{k: v if not isinstance(v, tuple) else list(v) for k, v in r.items()} 
                    for r in results]
    with open(output_dir / "explanation_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nAnalyse complète sauvegardée dans: {output_dir}")


if __name__ == "__main__":
    main()
