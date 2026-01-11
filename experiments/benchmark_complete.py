"""
Complete Benchmark Suite
========================
Script de benchmark complet comparant:
- DQN Vanilla, PER, Memory, Hierarchical, Full
- Baselines: A2C, PPO, SAC
- Sur plusieurs environnements avec reproductibilité

Usage:
    python experiments/benchmark_complete.py --env gridworld --episodes 1000 --seeds 5
    python experiments/benchmark_complete.py --env all --episodes 2000 --seeds 10

Auteur: ProRL Project
Date: 2025
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import time
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.reproducibility import (
    SeedManager, StatisticalAnalyzer, ExperimentLogger, 
    ExperimentConfig, ExperimentComparator
)


def get_state_vector(state, env_type: str) -> np.ndarray:
    """Convertit l'observation en vecteur numpy."""
    if isinstance(state, dict):
        if 'agent' in state:
            # GridWorld
            return state['agent'].astype(np.float32)
        elif 'state' in state:
            # Warehouse
            return state['state'].astype(np.float32)
        else:
            # Fallback - concatenate all arrays
            return np.concatenate([v.flatten() for v in state.values() if isinstance(v, np.ndarray)]).astype(np.float32)
    elif isinstance(state, np.ndarray):
        return state.astype(np.float32)
    else:
        return np.array(state, dtype=np.float32)


def create_environment(env_type: str, seed: int = 42):
    """Factory pour créer les environnements."""
    if env_type == 'gridworld':
        from env.gridworld import GridWorld
        return GridWorld()
    elif env_type == 'warehouse':
        from env.warehouse_robot import WarehouseEnv
        return WarehouseEnv()
    else:
        raise ValueError(f"Unknown environment: {env_type}")


def create_agent(agent_type: str, obs_dim: int, action_dim: int, seed: int = 42):
    """Factory pour créer les agents."""
    
    # Set seed
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    
    if agent_type == 'vanilla':
        from agents.dqn_base import DQNAgent
        return DQNAgent(obs_dim, action_dim, prioritized=False)
    
    elif agent_type == 'per':
        from agents.dqn_base import DQNAgent
        return DQNAgent(obs_dim, action_dim, prioritized=True)
    
    elif agent_type == 'memory':
        from agents.dqn_base import DQNAgent
        from agents.episodic_memory import EpisodicMemory
        agent = DQNAgent(obs_dim, action_dim, prioritized=False)
        agent.memory = EpisodicMemory(capacity=500)
        return agent
    
    elif agent_type == 'hier':
        from agents.hierarchical_dqn import HierarchicalDQN
        return HierarchicalDQN(obs_dim, action_dim, subgoal_dim=3)
    
    elif agent_type == 'full':
        from agents.hierarchical_dqn import HierarchicalDQN
        from agents.episodic_memory import EpisodicMemory
        agent = HierarchicalDQN(obs_dim, action_dim, subgoal_dim=3)
        agent.memory = EpisodicMemory(capacity=500)
        return agent
    
    elif agent_type == 'a2c':
        from agents.policy_gradients import A2C
        return A2C(obs_dim, action_dim, seed=seed)
    
    elif agent_type == 'ppo':
        from agents.policy_gradients import PPO
        return PPO(obs_dim, action_dim, seed=seed, rollout_length=256)
    
    elif agent_type == 'sac':
        from agents.policy_gradients import SACDiscrete
        return SACDiscrete(obs_dim, action_dim, seed=seed)
    
    else:
        raise ValueError(f"Unknown agent: {agent_type}")


def train_agent(
    env,
    agent,
    agent_type: str,
    n_episodes: int,
    env_type: str = 'gridworld',
    verbose: bool = False
) -> Dict:
    """Entraîne un agent et retourne les métriques."""
    
    episode_returns = []
    episode_lengths = []
    goals_reached = []
    
    start_time = time.time()
    
    for ep in range(n_episodes):
        state_raw = env.reset()
        state = get_state_vector(state_raw, env_type)
        total_reward = 0
        steps = 0
        done = False
        
        # Pour les agents hiérarchiques
        if agent_type == 'hier' or agent_type == 'full':
            subgoal = agent.select_subgoal(state, eps=max(0.1, 1.0 - ep/500))
        
        # Pour les agents policy gradient
        if agent_type in ['a2c', 'ppo']:
            log_probs = []
            values = []
        
        while not done and steps < 200:
            # Select action
            if agent_type in ['hier', 'full']:
                eps = max(0.05, 1.0 - ep/500)
                action = agent.select_action(state, subgoal, eps)
            elif agent_type in ['a2c', 'ppo']:
                action, log_prob, value = agent.select_action(state)
                log_probs.append(log_prob)
                values.append(value)
            elif agent_type == 'sac':
                action = agent.select_action(state)
            else:
                eps = max(0.05, 1.0 - ep/500)
                action = agent.select_action(state, eps) if hasattr(agent, 'select_action') else np.random.randint(env.action_dim)
            
            # Step
            next_state_raw, reward, done, info = env.step(action)
            next_state = get_state_vector(next_state_raw, env_type)
            
            # Store transition
            if agent_type in ['hier', 'full']:
                agent.push_low(state, subgoal, action, reward, next_state, done)
            elif agent_type == 'sac':
                agent.store_transition(state, action, reward, next_state, done)
            elif agent_type in ['a2c', 'ppo']:
                agent.store_transition(state, action, reward, done, log_probs[-1], values[-1])
            elif hasattr(agent, 'push'):
                agent.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # Update agent
        if agent_type in ['hier', 'full']:
            agent.update_low()
            if ep % 10 == 0:
                agent.update_high()
        elif agent_type == 'sac':
            for _ in range(steps):
                agent.update()
        elif agent_type in ['a2c', 'ppo']:
            agent.update(state)
        elif hasattr(agent, 'update'):
            for _ in range(4):
                agent.update()
        
        episode_returns.append(total_reward)
        episode_lengths.append(steps)
        goals_reached.append(1 if info.get('goal_reached', total_reward > 0) else 0)
        
        if verbose and (ep + 1) % 100 == 0:
            avg_return = np.mean(episode_returns[-100:])
            avg_goals = np.mean(goals_reached[-100:]) * 100
            print(f"    Episode {ep+1}/{n_episodes}: Return={avg_return:.1f}, Goals={avg_goals:.1f}%")
    
    training_time = time.time() - start_time
    
    # Compute metrics
    returns = np.array(episode_returns)
    metrics = {
        'mean_return': float(np.mean(returns)),
        'std_return': float(np.std(returns)),
        'final_mean_return': float(np.mean(returns[-100:])),
        'final_std_return': float(np.std(returns[-100:])),
        'max_return': float(np.max(returns)),
        'goal_rate': float(np.mean(goals_reached[-100:])),
        'training_time': training_time,
        'sample_efficiency': float(np.sum(returns > 0) / len(returns))
    }
    
    return {
        'metrics': metrics,
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
        'goals_reached': goals_reached
    }


def run_benchmark(
    env_types: List[str],
    agent_types: List[str],
    n_episodes: int,
    seeds: List[int],
    results_dir: str = "results/benchmark",
    verbose: bool = True
) -> Dict:
    """Exécute le benchmark complet."""
    
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    seed_manager = SeedManager()
    analyzer = StatisticalAnalyzer()
    
    all_results = {}
    
    total_runs = len(env_types) * len(agent_types) * len(seeds)
    current_run = 0
    
    for env_type in env_types:
        all_results[env_type] = {}
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Environment: {env_type.upper()}")
            print(f"{'='*60}")
        
        for agent_type in agent_types:
            all_results[env_type][agent_type] = {
                'runs': [],
                'summary': None
            }
            
            if verbose:
                print(f"\n  Agent: {agent_type}")
                print(f"  {'-'*40}")
            
            for seed in seeds:
                current_run += 1
                
                if verbose:
                    print(f"    Run {current_run}/{total_runs} (seed={seed})...")
                
                # Set seeds
                seed_manager.set_all_seeds(seed)
                
                # Create environment and agent
                try:
                    env = create_environment(env_type, seed)
                    agent = create_agent(agent_type, env.obs_dim, env.action_dim, seed)
                    
                    # Train
                    run_result = train_agent(env, agent, agent_type, n_episodes, env_type=env_type, verbose=False)
                    run_result['seed'] = seed
                    
                    all_results[env_type][agent_type]['runs'].append(run_result)
                    
                    if verbose:
                        print(f"      Final return: {run_result['metrics']['final_mean_return']:.2f}")
                        
                except Exception as e:
                    print(f"      ERROR: {e}")
                    continue
            
            # Compute summary statistics
            if all_results[env_type][agent_type]['runs']:
                final_returns = [r['metrics']['final_mean_return'] 
                                for r in all_results[env_type][agent_type]['runs']]
                goal_rates = [r['metrics']['goal_rate'] 
                             for r in all_results[env_type][agent_type]['runs']]
                
                from dataclasses import asdict
                all_results[env_type][agent_type]['summary'] = {
                    'final_return': asdict(analyzer.compute_summary(final_returns)),
                    'goal_rate': asdict(analyzer.compute_summary(goal_rates))
                }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full results (without episode data for size)
    summary_results = {}
    for env_type in all_results:
        summary_results[env_type] = {}
        for agent_type in all_results[env_type]:
            summary_results[env_type][agent_type] = {
                'n_runs': len(all_results[env_type][agent_type]['runs']),
                'summary': all_results[env_type][agent_type]['summary'],
                'per_seed_metrics': [
                    {'seed': r['seed'], 'metrics': r['metrics']}
                    for r in all_results[env_type][agent_type]['runs']
                ]
            }
    
    results_file = results_path / f"benchmark_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(summary_results, f, indent=2)
    
    if verbose:
        print(f"\nResults saved to: {results_file}")
    
    # Generate comparison report
    report = generate_report(summary_results)
    report_file = results_path / f"benchmark_report_{timestamp}.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    if verbose:
        print(f"Report saved to: {report_file}")
        print("\n" + report)
    
    return summary_results


def generate_report(results: Dict) -> str:
    """Génère un rapport de benchmark."""
    
    lines = [
        "# Benchmark Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    
    for env_type in results:
        lines.extend([
            f"## Environment: {env_type}",
            "",
            "### Performance Comparison",
            "",
            "| Agent | Final Return (Mean ± Std) | 95% CI | Goal Rate |",
            "|-------|---------------------------|--------|-----------|",
        ])
        
        # Sort by final return
        agents = list(results[env_type].keys())
        agents.sort(
            key=lambda a: results[env_type][a]['summary']['final_return']['mean'] 
            if results[env_type][a]['summary'] else -float('inf'),
            reverse=True
        )
        
        for agent in agents:
            data = results[env_type][agent]['summary']
            if data:
                mean = data['final_return']['mean']
                std = data['final_return']['std']
                ci_low = data['final_return']['ci_95_lower']
                ci_high = data['final_return']['ci_95_upper']
                goal = data['goal_rate']['mean'] * 100
                
                lines.append(
                    f"| {agent} | {mean:.2f} ± {std:.2f} | [{ci_low:.2f}, {ci_high:.2f}] | {goal:.1f}% |"
                )
        
        lines.extend(["", ""])
    
    # Key findings
    lines.extend([
        "## Key Findings",
        "",
    ])
    
    for env_type in results:
        agents_by_perf = sorted(
            [(k, v) for k, v in results[env_type].items() if v.get('summary')],
            key=lambda x: x[1]['summary']['final_return']['mean'] if x[1]['summary'] else -float('inf'),
            reverse=True
        )
        
        if agents_by_perf and agents_by_perf[0][1].get('summary'):
            best = agents_by_perf[0][0]
            best_perf = agents_by_perf[0][1]['summary']['final_return']['mean']
            lines.append(f"- **{env_type}**: Best agent = `{best}` (mean return: {best_perf:.2f})")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Complete DQN Benchmark Suite')
    parser.add_argument('--env', type=str, default='gridworld',
                       choices=['gridworld', 'warehouse', 'all'],
                       help='Environment to test')
    parser.add_argument('--agents', type=str, default='all',
                       help='Comma-separated list of agents or "all"')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes per run')
    parser.add_argument('--seeds', type=int, default=5,
                       help='Number of random seeds')
    parser.add_argument('--output', type=str, default='results/benchmark',
                       help='Output directory')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Environment selection
    if args.env == 'all':
        env_types = ['gridworld', 'warehouse']
    else:
        env_types = [args.env]
    
    # Agent selection
    all_agents = ['vanilla', 'per', 'memory', 'hier', 'full', 'a2c', 'ppo', 'sac']
    if args.agents == 'all':
        agent_types = all_agents
    else:
        agent_types = [a.strip() for a in args.agents.split(',')]
    
    # Seeds
    seeds = SeedManager.get_default_seeds(args.seeds)
    
    print("="*60)
    print("COMPLETE BENCHMARK SUITE")
    print("="*60)
    print(f"Environments: {env_types}")
    print(f"Agents: {agent_types}")
    print(f"Episodes: {args.episodes}")
    print(f"Seeds: {seeds}")
    print("="*60)
    
    results = run_benchmark(
        env_types=env_types,
        agent_types=agent_types,
        n_episodes=args.episodes,
        seeds=seeds,
        results_dir=args.output,
        verbose=not args.quiet
    )
    
    return results


if __name__ == "__main__":
    main()
