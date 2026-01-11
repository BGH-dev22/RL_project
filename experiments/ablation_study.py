"""
Ablation Studies Module
=======================
Analyse systématique de l'impact de chaque composant DQN.

Ce module implémente des études d'ablation rigoureuses pour:
1. Isoler l'impact de chaque composant (PER, Memory, Hierarchical)
2. Mesurer les interactions entre composants
3. Identifier les contributions marginales

Méthodologie:
- Configuration baseline (DQN vanilla)
- Ajout incrémental de composants
- Retrait sélectif de composants (Full - X)
- Statistiques sur plusieurs seeds

Auteur: ProRL Project
Date: 2025
"""

from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import time


@dataclass
class AblationConfig:
    """Configuration pour une expérience d'ablation."""
    name: str
    components: List[str]  # Liste des composants actifs
    description: str
    
    def __hash__(self):
        return hash((self.name, tuple(self.components)))


@dataclass 
class AblationResult:
    """Résultats d'une expérience d'ablation."""
    config_name: str
    seed: int
    metrics: Dict[str, float]
    training_time: float
    episode_returns: List[float]
    convergence_episode: Optional[int]


class AblationStudy:
    """
    Gestionnaire d'études d'ablation systématiques.
    
    Implémente:
    1. Leave-One-Out: Retirer un composant à la fois
    2. Incremental: Ajouter un composant à la fois
    3. Pairwise: Tester toutes les paires de composants
    """
    
    # Définition des composants
    ALL_COMPONENTS = ['prioritized_replay', 'episodic_memory', 'hierarchical', 'explainability']
    
    # Configurations d'ablation standard
    STANDARD_CONFIGS = {
        'vanilla': AblationConfig(
            name='vanilla',
            components=[],
            description='DQN baseline sans améliorations'
        ),
        'per_only': AblationConfig(
            name='per_only',
            components=['prioritized_replay'],
            description='DQN + Prioritized Experience Replay'
        ),
        'memory_only': AblationConfig(
            name='memory_only',
            components=['episodic_memory'],
            description='DQN + Mémoire Épisodique'
        ),
        'hier_only': AblationConfig(
            name='hier_only',
            components=['hierarchical'],
            description='DQN Hiérarchique seul'
        ),
        'per_memory': AblationConfig(
            name='per_memory',
            components=['prioritized_replay', 'episodic_memory'],
            description='PER + Mémoire Épisodique'
        ),
        'per_hier': AblationConfig(
            name='per_hier',
            components=['prioritized_replay', 'hierarchical'],
            description='PER + Hiérarchique'
        ),
        'memory_hier': AblationConfig(
            name='memory_hier',
            components=['episodic_memory', 'hierarchical'],
            description='Mémoire + Hiérarchique'
        ),
        'full': AblationConfig(
            name='full',
            components=['prioritized_replay', 'episodic_memory', 'hierarchical'],
            description='Toutes les améliorations combinées'
        ),
        'full_minus_per': AblationConfig(
            name='full_minus_per',
            components=['episodic_memory', 'hierarchical'],
            description='Full sans PER (Leave-One-Out)'
        ),
        'full_minus_memory': AblationConfig(
            name='full_minus_memory',
            components=['prioritized_replay', 'hierarchical'],
            description='Full sans Mémoire (Leave-One-Out)'
        ),
        'full_minus_hier': AblationConfig(
            name='full_minus_hier',
            components=['prioritized_replay', 'episodic_memory'],
            description='Full sans Hiérarchique (Leave-One-Out)'
        ),
        'full_explain': AblationConfig(
            name='full_explain',
            components=['prioritized_replay', 'episodic_memory', 'hierarchical', 'explainability'],
            description='Full + Explainability'
        )
    }
    
    def __init__(
        self,
        env_factory: Callable,
        agent_factory: Callable,
        n_episodes: int = 1000,
        seeds: List[int] = [42, 123, 456, 789, 1011],
        results_dir: str = "results/ablation",
        verbose: bool = True
    ):
        """
        Args:
            env_factory: Fonction () -> env pour créer l'environnement
            agent_factory: Fonction (config, seed) -> agent pour créer l'agent
            n_episodes: Nombre d'épisodes par run
            seeds: Liste de seeds pour reproductibilité
            results_dir: Répertoire de sauvegarde
            verbose: Afficher le progrès
        """
        self.env_factory = env_factory
        self.agent_factory = agent_factory
        self.n_episodes = n_episodes
        self.seeds = seeds
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        self.results: Dict[str, List[AblationResult]] = {}
        
    def run_single_experiment(
        self,
        config: AblationConfig,
        seed: int
    ) -> AblationResult:
        """Exécute une seule expérience d'ablation."""
        if self.verbose:
            print(f"  Running {config.name} with seed {seed}...")
        
        # Set seeds
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass
        
        # Create environment and agent
        env = self.env_factory()
        agent = self.agent_factory(config, seed)
        
        # Training
        start_time = time.time()
        episode_returns = []
        convergence_episode = None
        convergence_threshold = 0.7  # 70% goal rate
        
        for ep in range(self.n_episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                agent.update()
                state = next_state
                total_reward += reward
            
            episode_returns.append(total_reward)
            
            # Check convergence
            if convergence_episode is None and len(episode_returns) >= 100:
                recent_avg = np.mean(episode_returns[-100:])
                if recent_avg > 0:  # Positive return indicates success
                    convergence_episode = ep
        
        training_time = time.time() - start_time
        
        # Compute metrics
        metrics = self._compute_metrics(episode_returns, env)
        
        return AblationResult(
            config_name=config.name,
            seed=seed,
            metrics=metrics,
            training_time=training_time,
            episode_returns=episode_returns,
            convergence_episode=convergence_episode
        )
    
    def _compute_metrics(
        self,
        episode_returns: List[float],
        env
    ) -> Dict[str, float]:
        """Calcule les métriques de performance."""
        returns = np.array(episode_returns)
        
        # Last 100 episodes
        last_100 = returns[-100:]
        
        metrics = {
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'final_mean_return': float(np.mean(last_100)),
            'final_std_return': float(np.std(last_100)),
            'max_return': float(np.max(returns)),
            'min_return': float(np.min(returns)),
            'success_rate': float(np.mean(last_100 > 0)),
            'sample_efficiency': self._compute_sample_efficiency(returns)
        }
        
        return metrics
    
    def _compute_sample_efficiency(self, returns: np.ndarray) -> float:
        """Calcule l'efficacité d'échantillonnage (AUC normalisé)."""
        # Area under the learning curve, normalized
        cumulative = np.cumsum(returns)
        max_possible = len(returns) * np.max(returns)
        if max_possible > 0:
            return float(cumulative[-1] / max_possible)
        return 0.0
    
    def run_ablation_study(
        self,
        configs: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Exécute l'étude d'ablation complète.
        
        Args:
            configs: Liste des noms de configurations à tester.
                    Si None, teste toutes les configurations standard.
        """
        if configs is None:
            configs = list(self.STANDARD_CONFIGS.keys())
        
        if self.verbose:
            print(f"Starting ablation study with {len(configs)} configs and {len(self.seeds)} seeds")
            print(f"Total experiments: {len(configs) * len(self.seeds)}")
        
        for config_name in configs:
            config = self.STANDARD_CONFIGS[config_name]
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Configuration: {config.name}")
                print(f"Components: {config.components}")
                print(f"{'='*50}")
            
            self.results[config_name] = []
            
            for seed in self.seeds:
                result = self.run_single_experiment(config, seed)
                self.results[config_name].append(result)
        
        # Aggregate and save results
        summary = self._aggregate_results()
        self._save_results(summary)
        
        return summary
    
    def _aggregate_results(self) -> Dict[str, Dict]:
        """Agrège les résultats sur tous les seeds."""
        summary = {}
        
        for config_name, results in self.results.items():
            config = self.STANDARD_CONFIGS[config_name]
            
            # Aggregate metrics
            all_metrics = {}
            for key in results[0].metrics.keys():
                values = [r.metrics[key] for r in results]
                all_metrics[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'ci_95': float(1.96 * np.std(values) / np.sqrt(len(values))),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
            
            # Convergence statistics
            conv_episodes = [r.convergence_episode for r in results if r.convergence_episode is not None]
            
            summary[config_name] = {
                'config': asdict(config),
                'n_seeds': len(results),
                'metrics': all_metrics,
                'convergence': {
                    'converged_runs': len(conv_episodes),
                    'mean_episode': float(np.mean(conv_episodes)) if conv_episodes else None,
                    'std_episode': float(np.std(conv_episodes)) if conv_episodes else None
                },
                'training_time': {
                    'mean': float(np.mean([r.training_time for r in results])),
                    'std': float(np.std([r.training_time for r in results]))
                }
            }
        
        return summary
    
    def _save_results(self, summary: Dict):
        """Sauvegarde les résultats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary
        summary_path = self.results_dir / f"ablation_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        detailed = {}
        for config_name, results in self.results.items():
            detailed[config_name] = [
                {
                    'seed': r.seed,
                    'metrics': r.metrics,
                    'training_time': r.training_time,
                    'convergence_episode': r.convergence_episode,
                    'final_returns': r.episode_returns[-100:]
                }
                for r in results
            ]
        
        detailed_path = self.results_dir / f"ablation_detailed_{timestamp}.json"
        with open(detailed_path, 'w') as f:
            json.dump(detailed, f, indent=2)
        
        if self.verbose:
            print(f"\nResults saved to:")
            print(f"  Summary: {summary_path}")
            print(f"  Detailed: {detailed_path}")
    
    def compute_component_importance(self) -> Dict[str, Dict[str, float]]:
        """
        Calcule l'importance de chaque composant via Shapley-like analysis.
        
        Pour chaque composant, mesure:
        1. Contribution marginale (Full - Full_minus_X)
        2. Impact isolé (X_only - vanilla)
        3. Importance relative (normalised)
        """
        if not self.results:
            raise ValueError("Run ablation study first")
        
        importance = {}
        baseline_metric = self.results['vanilla'][0].metrics['final_mean_return']
        full_metric = self.results['full'][0].metrics['final_mean_return']
        
        for component in ['per', 'memory', 'hier']:
            component_full = f'full_minus_{component}'
            component_only = f'{component}_only'
            
            if component_full in self.results and component_only in self.results:
                full_minus = np.mean([r.metrics['final_mean_return'] 
                                     for r in self.results[component_full]])
                only = np.mean([r.metrics['final_mean_return'] 
                              for r in self.results[component_only]])
                
                # Marginal contribution
                marginal = full_metric - full_minus
                
                # Isolated impact
                isolated = only - baseline_metric
                
                # Relative importance
                total_improvement = full_metric - baseline_metric
                relative = marginal / total_improvement if total_improvement > 0 else 0
                
                importance[component] = {
                    'marginal_contribution': float(marginal),
                    'isolated_impact': float(isolated),
                    'relative_importance': float(relative),
                    'synergy_factor': float(marginal / isolated) if isolated != 0 else 1.0
                }
        
        return importance
    
    def generate_report(self) -> str:
        """Génère un rapport textuel de l'étude d'ablation."""
        if not self.results:
            return "No results available. Run ablation study first."
        
        summary = self._aggregate_results()
        importance = self.compute_component_importance()
        
        lines = [
            "=" * 60,
            "ABLATION STUDY REPORT",
            "=" * 60,
            "",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Number of configurations: {len(summary)}",
            f"Seeds used: {self.seeds}",
            f"Episodes per run: {self.n_episodes}",
            "",
            "-" * 60,
            "PERFORMANCE SUMMARY",
            "-" * 60,
            ""
        ]
        
        # Performance table
        lines.append(f"{'Config':<20} {'Mean Return':<15} {'95% CI':<15} {'Success Rate':<15}")
        lines.append("-" * 65)
        
        for config_name, data in sorted(summary.items(), 
                                        key=lambda x: x[1]['metrics']['final_mean_return']['mean'],
                                        reverse=True):
            mean = data['metrics']['final_mean_return']['mean']
            ci = data['metrics']['final_mean_return']['ci_95']
            success = data['metrics']['success_rate']['mean']
            lines.append(f"{config_name:<20} {mean:>10.2f}     ±{ci:>6.2f}        {success*100:>6.1f}%")
        
        lines.extend([
            "",
            "-" * 60,
            "COMPONENT IMPORTANCE ANALYSIS",
            "-" * 60,
            ""
        ])
        
        for component, data in importance.items():
            lines.append(f"\n{component.upper()}:")
            lines.append(f"  Marginal contribution: {data['marginal_contribution']:+.2f}")
            lines.append(f"  Isolated impact: {data['isolated_impact']:+.2f}")
            lines.append(f"  Relative importance: {data['relative_importance']*100:.1f}%")
            lines.append(f"  Synergy factor: {data['synergy_factor']:.2f}x")
        
        lines.extend([
            "",
            "-" * 60,
            "KEY FINDINGS",
            "-" * 60,
        ])
        
        # Find best and worst
        best = max(summary.items(), key=lambda x: x[1]['metrics']['final_mean_return']['mean'])
        worst = min(summary.items(), key=lambda x: x[1]['metrics']['final_mean_return']['mean'])
        
        lines.append(f"\n• Best configuration: {best[0]} (mean return: {best[1]['metrics']['final_mean_return']['mean']:.2f})")
        lines.append(f"• Worst configuration: {worst[0]} (mean return: {worst[1]['metrics']['final_mean_return']['mean']:.2f})")
        
        # Most important component
        most_important = max(importance.items(), key=lambda x: x[1]['relative_importance'])
        lines.append(f"• Most important component: {most_important[0]} ({most_important[1]['relative_importance']*100:.1f}% relative importance)")
        
        # Best synergy
        best_synergy = max(importance.items(), key=lambda x: x[1]['synergy_factor'])
        lines.append(f"• Strongest synergy: {best_synergy[0]} ({best_synergy[1]['synergy_factor']:.2f}x synergy factor)")
        
        return "\n".join(lines)


# =============================================================================
# SIMPLIFIED ABLATION RUNNER
# =============================================================================

def run_quick_ablation(n_episodes: int = 500, seeds: List[int] = [42, 123, 456]):
    """
    Lance une étude d'ablation rapide avec les paramètres par défaut.
    
    Usage:
        python -c "from experiments.ablation_study import run_quick_ablation; run_quick_ablation()"
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from env.gridworld import GridWorld
    from agents.dqn_base import DQNAgent
    from agents.hierarchical_dqn import HierarchicalDQN
    from agents.episodic_memory import EpisodicMemory
    
    def env_factory():
        return GridWorld()
    
    def agent_factory(config: AblationConfig, seed: int):
        # Simplified agent creation based on config
        env = GridWorld()
        obs_dim = env.obs_dim
        action_dim = env.action_dim
        
        if 'hierarchical' in config.components:
            agent = HierarchicalDQN(obs_dim, action_dim, subgoal_dim=3)
        else:
            prioritized = 'prioritized_replay' in config.components
            agent = DQNAgent(obs_dim, action_dim, prioritized=prioritized)
        
        return agent
    
    study = AblationStudy(
        env_factory=env_factory,
        agent_factory=agent_factory,
        n_episodes=n_episodes,
        seeds=seeds
    )
    
    # Run subset of configurations for quick test
    configs = ['vanilla', 'per_only', 'memory_only', 'hier_only', 'full']
    summary = study.run_ablation_study(configs)
    
    print("\n" + study.generate_report())
    
    return summary


if __name__ == "__main__":
    run_quick_ablation()
