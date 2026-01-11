"""
Transfer Learning Module for DQN
================================
INNOVATION: Démontrer que les skills appris se transfèrent à de nouveaux environnements.

Includes:
1. Skill extraction from trained agents
2. Transfer to modified environments
3. Fine-tuning with frozen/unfrozen layers
4. Zero-shot and few-shot transfer evaluation

Auteur: ProRL Project
Date: 2025
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import copy
from pathlib import Path
import json


class SkillExtractor:
    """
    Extract reusable skills from trained DQN agents.
    
    INNOVATION: Skills are represented as:
    1. Feature extractors (lower layers)
    2. Subgoal policies (for hierarchical agents)
    3. Episode patterns (from episodic memory)
    """
    
    def __init__(self, agent, agent_type: str = 'dqn'):
        self.agent = agent
        self.agent_type = agent_type
        self.extracted_skills = {}
        
    def extract_feature_layers(self) -> nn.Module:
        """Extract feature extraction layers (transferable)."""
        if self.agent_type == 'hierarchical':
            # For hierarchical, extract both high and low level features
            return {
                'low_features': self._copy_layers(self.agent.low_level, num_layers=2),
                'high_features': self._copy_layers(self.agent.high_level.net.net, num_layers=2)
            }
        else:
            # For standard DQN
            return self._copy_layers(self.agent.policy.net, num_layers=2)
    
    def _copy_layers(self, network: nn.Sequential, num_layers: int) -> nn.Module:
        """Copy first n layers from a network."""
        layers = []
        count = 0
        for layer in network:
            layers.append(copy.deepcopy(layer))
            if isinstance(layer, nn.Linear):
                count += 1
                if count >= num_layers:
                    break
        return nn.Sequential(*layers)
    
    def extract_subgoal_policy(self) -> Optional[nn.Module]:
        """Extract subgoal selection policy (for hierarchical agents)."""
        if self.agent_type != 'hierarchical':
            return None
        return copy.deepcopy(self.agent.high_level)
    
    def extract_action_patterns(self, episodes: List) -> Dict[str, List[int]]:
        """Extract common action patterns from successful episodes."""
        patterns = {
            'to_key': [],
            'to_door': [],
            'to_goal': []
        }
        
        for ep in episodes:
            if not hasattr(ep, 'subgoals_achieved'):
                continue
                
            trajectory = ep.trajectory
            
            # Find transitions for each subgoal
            key_idx = None
            door_idx = None
            
            for i, (s, a, r) in enumerate(trajectory):
                if 'key' in ep.subgoals_achieved and key_idx is None:
                    if s[2] == 1:  # has_key became true
                        patterns['to_key'].extend([a for _, a, _ in trajectory[:i+1]])
                        key_idx = i
                
                if 'door' in ep.subgoals_achieved and door_idx is None and key_idx is not None:
                    if r > 15:  # door opened bonus
                        patterns['to_door'].extend([a for _, a, _ in trajectory[key_idx:i+1]])
                        door_idx = i
                
                if 'goal' in ep.subgoals_achieved and door_idx is not None:
                    if r > 90:  # goal reached
                        patterns['to_goal'].extend([a for _, a, _ in trajectory[door_idx:i+1]])
        
        return patterns
    
    def save_skills(self, path: str) -> None:
        """Save extracted skills to disk."""
        skills_dir = Path(path)
        skills_dir.mkdir(parents=True, exist_ok=True)
        
        # Save feature extractors
        features = self.extract_feature_layers()
        if isinstance(features, dict):
            for name, module in features.items():
                torch.save(module.state_dict(), skills_dir / f"{name}.pt")
        else:
            torch.save(features.state_dict(), skills_dir / "features.pt")
        
        # Save subgoal policy if available
        subgoal_policy = self.extract_subgoal_policy()
        if subgoal_policy is not None:
            torch.save(subgoal_policy.state_dict(), skills_dir / "subgoal_policy.pt")
        
        print(f"Skills saved to {skills_dir}")


class TransferableGridWorld:
    """
    Modified GridWorld environments for transfer learning evaluation.
    
    INNOVATION: Multiple environment variations to test generalization:
    1. Larger grid
    2. Multiple keys
    3. Multiple doors
    4. Different layouts
    """
    
    def __init__(self, variation: str = 'larger', base_size: int = 10):
        self.variation = variation
        self.base_size = base_size
        self.size = self._get_size()
        self.max_steps = self._get_max_steps()
        self._setup_environment()
    
    def _get_size(self) -> int:
        sizes = {
            'larger': self.base_size + 5,
            'smaller': max(6, self.base_size - 3),
            'multi_key': self.base_size,
            'multi_door': self.base_size + 3,
            'maze': self.base_size + 2
        }
        return sizes.get(self.variation, self.base_size)
    
    def _get_max_steps(self) -> int:
        return int(500 * (self.size / self.base_size) ** 1.5)
    
    def _setup_environment(self) -> None:
        """Setup environment based on variation."""
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.walls = []
        self.traps = []
        self.keys = []
        self.doors = []
        
        if self.variation == 'larger':
            self._setup_larger()
        elif self.variation == 'smaller':
            self._setup_smaller()
        elif self.variation == 'multi_key':
            self._setup_multi_key()
        elif self.variation == 'multi_door':
            self._setup_multi_door()
        elif self.variation == 'maze':
            self._setup_maze()
        else:
            self._setup_default()
    
    def _setup_larger(self) -> None:
        """Larger grid with same structure."""
        mid = self.size // 2
        self.walls = [(mid, j) for j in range(self.size) if j != mid]
        self.doors = [(mid, mid)]
        self.goal = (self.size - 2, self.size - 2)
        self.keys = [(1, self.size - 2)]
        self.traps = [(3, 3), (mid - 1, 2), (mid + 3, mid + 3)]
    
    def _setup_smaller(self) -> None:
        """Smaller, more compact grid."""
        mid = self.size // 2
        self.walls = [(mid, j) for j in range(self.size) if j != mid]
        self.doors = [(mid, mid)]
        self.goal = (self.size - 2, self.size - 2)
        self.keys = [(1, self.size - 2)]
        self.traps = [(2, 2)]
    
    def _setup_multi_key(self) -> None:
        """Multiple keys required."""
        mid = self.size // 2
        self.walls = [(mid, j) for j in range(self.size) if j not in [mid-1, mid, mid+1]]
        self.doors = [(mid, mid-1), (mid, mid), (mid, mid+1)]
        self.goal = (self.size - 2, self.size - 2)
        self.keys = [(1, 1), (1, self.size - 2), (mid - 2, mid)]
        self.traps = [(2, 2), (mid - 1, 1)]
        self.required_keys = 3
    
    def _setup_multi_door(self) -> None:
        """Multiple doors in sequence."""
        barriers = [self.size // 3, 2 * self.size // 3]
        for b in barriers:
            self.walls.extend([(b, j) for j in range(self.size) if j != self.size // 2])
            self.doors.append((b, self.size // 2))
        self.goal = (self.size - 2, self.size - 2)
        self.keys = [(1, 1), (barriers[0] + 1, 1)]
        self.traps = [(2, 2), (barriers[0] + 2, 2)]
    
    def _setup_maze(self) -> None:
        """Maze-like structure."""
        # Create maze walls
        for i in range(2, self.size - 2, 3):
            for j in range(self.size):
                if j != (i % self.size):
                    self.walls.append((i, j))
        self.doors = [(2, 2), (5, 5)]
        self.goal = (self.size - 2, self.size - 2)
        self.keys = [(1, 1), (3, 3)]
        self.traps = [(4, 4)]
    
    def _setup_default(self) -> None:
        """Default setup similar to original."""
        mid = self.size // 2
        self.walls = [(mid, j) for j in range(self.size) if j != mid]
        self.doors = [(mid, mid)]
        self.goal = (self.size - 2, self.size - 2)
        self.keys = [(1, self.size - 2)]
        self.traps = [(2, 2)]
    
    def get_description(self) -> Dict:
        """Get environment description for logging."""
        return {
            'variation': self.variation,
            'size': self.size,
            'num_walls': len(self.walls),
            'num_doors': len(self.doors) if hasattr(self, 'doors') else 1,
            'num_keys': len(self.keys),
            'num_traps': len(self.traps),
            'max_steps': self.max_steps
        }


class TransferLearningExperiment:
    """
    INNOVATION: Comprehensive transfer learning evaluation.
    
    Tests:
    1. Zero-shot transfer (direct application)
    2. Few-shot transfer (minimal fine-tuning)
    3. Full fine-tuning (with frozen features)
    4. Skill composition (combining learned skills)
    """
    
    def __init__(self, source_agent, source_type: str = 'dqn'):
        self.source_agent = source_agent
        self.source_type = source_type
        self.skill_extractor = SkillExtractor(source_agent, source_type)
        self.results = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def zero_shot_transfer(self, target_env, num_episodes: int = 100) -> Dict:
        """
        Evaluate zero-shot transfer (no fine-tuning).
        Tests if learned policy generalizes directly.
        """
        results = {
            'episodes': [],
            'returns': [],
            'goals_reached': 0,
            'keys_collected': 0,
            'doors_opened': 0
        }
        
        for ep in range(num_episodes):
            obs = target_env.reset()
            state = self._obs_to_state(obs)
            total_reward = 0
            done = False
            
            while not done:
                # Use source agent directly
                if self.source_type == 'hierarchical':
                    subgoal = self.source_agent.select_subgoal(state, eps=0.05)
                    action = self.source_agent.select_action(state, subgoal, eps=0.05)
                else:
                    action = self.source_agent.select_action(state, eps=0.05)
                
                obs, reward, done, info = target_env.step(action)
                state = self._obs_to_state(obs)
                total_reward += reward
                
                if info.get('got_key'):
                    results['keys_collected'] += 1
                if info.get('door_opened'):
                    results['doors_opened'] += 1
                if info.get('goal_reached'):
                    results['goals_reached'] += 1
            
            results['returns'].append(total_reward)
        
        results['mean_return'] = np.mean(results['returns'])
        results['goal_rate'] = results['goals_reached'] / num_episodes
        
        return results
    
    def few_shot_transfer(self, target_env, num_episodes: int = 100, 
                          fine_tune_episodes: int = 50) -> Dict:
        """
        Few-shot transfer with minimal fine-tuning.
        Freezes feature layers, only trains output layer.
        """
        # Create new agent with frozen features
        transferred_agent = self._create_transferred_agent(freeze_features=True)
        
        # Fine-tune on target environment
        fine_tune_results = self._fine_tune(
            transferred_agent, target_env, fine_tune_episodes
        )
        
        # Evaluate
        eval_results = self._evaluate_agent(
            transferred_agent, target_env, num_episodes
        )
        
        return {
            'fine_tune_results': fine_tune_results,
            'eval_results': eval_results,
            'fine_tune_episodes': fine_tune_episodes
        }
    
    def full_transfer(self, target_env, num_episodes: int = 100,
                      fine_tune_episodes: int = 200) -> Dict:
        """
        Full fine-tuning with feature initialization.
        All layers trainable but initialized from source.
        """
        transferred_agent = self._create_transferred_agent(freeze_features=False)
        
        fine_tune_results = self._fine_tune(
            transferred_agent, target_env, fine_tune_episodes
        )
        
        eval_results = self._evaluate_agent(
            transferred_agent, target_env, num_episodes
        )
        
        return {
            'fine_tune_results': fine_tune_results,
            'eval_results': eval_results,
            'fine_tune_episodes': fine_tune_episodes
        }
    
    def scratch_baseline(self, target_env, num_episodes: int = 100,
                         train_episodes: int = 200) -> Dict:
        """
        Baseline: Train from scratch on target environment.
        Used to measure transfer benefit.
        """
        from agents.dqn_base import DQNAgent
        
        # Create fresh agent
        obs = target_env.reset()
        state = self._obs_to_state(obs)
        fresh_agent = DQNAgent(
            obs_dim=len(state),
            action_dim=target_env.action_space,
            prioritized=True
        )
        
        # Train from scratch
        train_results = self._fine_tune(
            fresh_agent, target_env, train_episodes, is_scratch=True
        )
        
        # Evaluate
        eval_results = self._evaluate_agent(
            fresh_agent, target_env, num_episodes
        )
        
        return {
            'train_results': train_results,
            'eval_results': eval_results,
            'train_episodes': train_episodes
        }
    
    def _create_transferred_agent(self, freeze_features: bool) -> 'DQNAgent':
        """Create new agent with transferred weights."""
        from agents.dqn_base import DQNAgent
        
        # Get dimensions from source
        if self.source_type == 'hierarchical':
            obs_dim = self.source_agent.low_level[0].in_features - self.source_agent.subgoal_dim
            action_dim = self.source_agent.action_dim
        else:
            obs_dim = self.source_agent.policy.net[0].in_features
            action_dim = self.source_agent.policy.net[-1].out_features
        
        # Create new agent
        new_agent = DQNAgent(obs_dim=obs_dim, action_dim=action_dim, prioritized=True)
        
        # Copy weights from source
        if self.source_type != 'hierarchical':
            new_agent.policy.load_state_dict(self.source_agent.policy.state_dict())
            new_agent.target.load_state_dict(self.source_agent.target.state_dict())
        
        # Freeze features if requested
        if freeze_features:
            for i, layer in enumerate(new_agent.policy.net):
                if isinstance(layer, nn.Linear) and i < 2:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        return new_agent
    
    def _fine_tune(self, agent, env, num_episodes: int, 
                   is_scratch: bool = False) -> Dict:
        """Fine-tune agent on environment."""
        results = {'returns': [], 'goals': 0}
        
        eps_start = 1.0 if is_scratch else 0.3
        eps_end = 0.05
        eps_decay = (eps_start - eps_end) / max(1, num_episodes)
        
        for ep in range(num_episodes):
            eps = max(eps_end, eps_start - ep * eps_decay)
            obs = env.reset()
            state = self._obs_to_state(obs)
            total_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(state, eps)
                obs, reward, done, info = env.step(action)
                next_state = self._obs_to_state(obs)
                
                agent.push(state, action, reward, next_state, done)
                agent.update()
                
                state = next_state
                total_reward += reward
                
                if info.get('goal_reached'):
                    results['goals'] += 1
            
            results['returns'].append(total_reward)
        
        results['mean_return'] = np.mean(results['returns'])
        results['goal_rate'] = results['goals'] / num_episodes
        
        return results
    
    def _evaluate_agent(self, agent, env, num_episodes: int) -> Dict:
        """Evaluate agent without training."""
        results = {'returns': [], 'goals': 0}
        
        for ep in range(num_episodes):
            obs = env.reset()
            state = self._obs_to_state(obs)
            total_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(state, eps=0.05)
                obs, reward, done, info = env.step(action)
                state = self._obs_to_state(obs)
                total_reward += reward
                
                if info.get('goal_reached'):
                    results['goals'] += 1
            
            results['returns'].append(total_reward)
        
        results['mean_return'] = np.mean(results['returns'])
        results['goal_rate'] = results['goals'] / num_episodes
        
        return results
    
    def _obs_to_state(self, obs: Dict) -> np.ndarray:
        """Convert observation to state vector."""
        agent = obs['agent']
        return agent.astype(np.float32)
    
    def run_full_experiment(self, variations: List[str] = None) -> Dict:
        """Run complete transfer learning experiment."""
        if variations is None:
            variations = ['larger', 'smaller', 'multi_key', 'maze']
        
        all_results = {}
        
        for variation in variations:
            print(f"\n{'='*50}")
            print(f"Testing transfer to: {variation}")
            print('='*50)
            
            # Create target environment
            from env.gridworld import GridWorld
            target_env = GridWorld(
                size=TransferableGridWorld(variation).size,
                max_steps=TransferableGridWorld(variation).max_steps
            )
            
            # Run all transfer types
            variation_results = {}
            
            print("  Zero-shot transfer...")
            variation_results['zero_shot'] = self.zero_shot_transfer(target_env)
            print(f"    Goal rate: {variation_results['zero_shot']['goal_rate']:.1%}")
            
            print("  Few-shot transfer (50 episodes)...")
            variation_results['few_shot'] = self.few_shot_transfer(target_env)
            print(f"    Goal rate: {variation_results['few_shot']['eval_results']['goal_rate']:.1%}")
            
            print("  Full transfer (200 episodes)...")
            variation_results['full_transfer'] = self.full_transfer(target_env)
            print(f"    Goal rate: {variation_results['full_transfer']['eval_results']['goal_rate']:.1%}")
            
            print("  Training from scratch (200 episodes)...")
            variation_results['scratch'] = self.scratch_baseline(target_env)
            print(f"    Goal rate: {variation_results['scratch']['eval_results']['goal_rate']:.1%}")
            
            # Compute transfer benefit
            transfer_benefit = (
                variation_results['full_transfer']['eval_results']['goal_rate'] -
                variation_results['scratch']['eval_results']['goal_rate']
            )
            variation_results['transfer_benefit'] = transfer_benefit
            print(f"  Transfer benefit: {transfer_benefit:+.1%}")
            
            all_results[variation] = variation_results
        
        self.results = all_results
        return all_results
    
    def generate_report(self) -> str:
        """Generate transfer learning report."""
        report = []
        report.append("=" * 70)
        report.append("TRANSFER LEARNING EXPERIMENT RESULTS")
        report.append("=" * 70)
        
        for variation, results in self.results.items():
            report.append(f"\n{variation.upper()} ENVIRONMENT")
            report.append("-" * 40)
            
            report.append(f"  Zero-shot transfer:")
            report.append(f"    Goal rate: {results['zero_shot']['goal_rate']:.1%}")
            report.append(f"    Mean return: {results['zero_shot']['mean_return']:.1f}")
            
            report.append(f"  Few-shot transfer (50 ep fine-tune):")
            report.append(f"    Goal rate: {results['few_shot']['eval_results']['goal_rate']:.1%}")
            
            report.append(f"  Full transfer (200 ep fine-tune):")
            report.append(f"    Goal rate: {results['full_transfer']['eval_results']['goal_rate']:.1%}")
            
            report.append(f"  From scratch (200 ep):")
            report.append(f"    Goal rate: {results['scratch']['eval_results']['goal_rate']:.1%}")
            
            report.append(f"  TRANSFER BENEFIT: {results['transfer_benefit']:+.1%}")
        
        # Summary
        report.append("\n" + "=" * 70)
        report.append("SUMMARY")
        report.append("=" * 70)
        
        avg_zero_shot = np.mean([r['zero_shot']['goal_rate'] for r in self.results.values()])
        avg_transfer = np.mean([r['full_transfer']['eval_results']['goal_rate'] for r in self.results.values()])
        avg_scratch = np.mean([r['scratch']['eval_results']['goal_rate'] for r in self.results.values()])
        avg_benefit = np.mean([r['transfer_benefit'] for r in self.results.values()])
        
        report.append(f"Average zero-shot performance: {avg_zero_shot:.1%}")
        report.append(f"Average transfer performance: {avg_transfer:.1%}")
        report.append(f"Average from-scratch performance: {avg_scratch:.1%}")
        report.append(f"Average transfer benefit: {avg_benefit:+.1%}")
        
        report.append("\nCONCLUSION:")
        if avg_benefit > 0.05:
            report.append("✓ Transfer learning provides significant benefit!")
            report.append("✓ Learned skills successfully transfer to new environments.")
        elif avg_benefit > 0:
            report.append("○ Transfer learning provides modest benefit.")
            report.append("○ Some skills transfer, but adaptation is needed.")
        else:
            report.append("✗ Transfer learning shows no benefit.")
            report.append("✗ Skills do not generalize to tested environments.")
        
        return "\n".join(report)
    
    def save_results(self, output_dir: str = "results") -> None:
        """Save experiment results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(output_path / "transfer_learning_results.json", 'w') as f:
            # Convert numpy types
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                return obj
            
            json.dump(self.results, f, indent=2, default=convert)
        
        # Save report
        report = self.generate_report()
        with open(output_path / "transfer_learning_report.txt", 'w') as f:
            f.write(report)
        
        print(f"\nResults saved to {output_path}")


def run_transfer_experiment(agent, agent_type: str = 'dqn'):
    """Convenience function to run transfer experiment."""
    experiment = TransferLearningExperiment(agent, agent_type)
    results = experiment.run_full_experiment()
    print(experiment.generate_report())
    experiment.save_results()
    return experiment


if __name__ == "__main__":
    print("Transfer Learning module loaded.")
    print("Use run_transfer_experiment(agent) to run experiments.")
