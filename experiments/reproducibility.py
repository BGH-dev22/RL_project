"""
Reproducibility Module
======================
Outils pour assurer la reproductibilité des expériences:
- Gestion des seeds
- Intervalles de confiance
- Tests statistiques
- Logging des hyperparamètres

Auteur: ProRL Project
Date: 2025
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import datetime
import hashlib
import os
import sys


# =============================================================================
# SEED MANAGEMENT
# =============================================================================

class SeedManager:
    """
    Gestionnaire centralisé des seeds pour reproductibilité totale.
    
    Gère:
    - numpy random
    - Python random
    - PyTorch (CPU et CUDA)
    - Environnements Gymnasium
    """
    
    DEFAULT_SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
    
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.current_seed = base_seed
        self.seed_history = []
        
    def set_all_seeds(self, seed: int):
        """Configure tous les générateurs aléatoires avec le même seed."""
        self.current_seed = seed
        self.seed_history.append({
            'seed': seed,
            'timestamp': datetime.now().isoformat()
        })
        
        # Python random
        import random
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # PyTorch (si disponible)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                # Pour reproductibilité CUDA complète (peut réduire les performances)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
        
        # TensorFlow (si disponible)
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass
        
        return seed
    
    def get_seed_sequence(self, n_seeds: int) -> List[int]:
        """Génère une séquence de seeds reproductible."""
        np.random.seed(self.base_seed)
        seeds = np.random.randint(0, 2**31, size=n_seeds).tolist()
        # Reset to current
        np.random.seed(self.current_seed)
        return seeds
    
    @classmethod
    def get_default_seeds(cls, n: int = 5) -> List[int]:
        """Retourne les seeds par défaut pour expériences standard."""
        return cls.DEFAULT_SEEDS[:n]


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

@dataclass
class StatisticalSummary:
    """Résumé statistique d'une série de mesures."""
    n: int
    mean: float
    std: float
    sem: float  # Standard Error of Mean
    ci_95_lower: float
    ci_95_upper: float
    ci_99_lower: float
    ci_99_upper: float
    median: float
    q25: float
    q75: float
    min: float
    max: float
    iqr: float  # Interquartile Range


class StatisticalAnalyzer:
    """
    Analyseur statistique pour résultats d'expériences.
    
    Fournit:
    - Intervalles de confiance (95%, 99%)
    - Tests de significativité
    - Comparaisons multiples avec correction
    """
    
    @staticmethod
    def compute_summary(data: List[float]) -> StatisticalSummary:
        """Calcule un résumé statistique complet."""
        arr = np.array(data)
        n = len(arr)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)  # Sample std
        sem = std / np.sqrt(n)
        
        # t-value for 95% and 99% CI (approximation for large n)
        t_95 = 1.96 if n > 30 else 2.0  # Simplified
        t_99 = 2.576 if n > 30 else 2.7
        
        return StatisticalSummary(
            n=n,
            mean=float(mean),
            std=float(std),
            sem=float(sem),
            ci_95_lower=float(mean - t_95 * sem),
            ci_95_upper=float(mean + t_95 * sem),
            ci_99_lower=float(mean - t_99 * sem),
            ci_99_upper=float(mean + t_99 * sem),
            median=float(np.median(arr)),
            q25=float(np.percentile(arr, 25)),
            q75=float(np.percentile(arr, 75)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            iqr=float(np.percentile(arr, 75) - np.percentile(arr, 25))
        )
    
    @staticmethod
    def welch_t_test(group1: List[float], group2: List[float]) -> Dict[str, float]:
        """
        Welch's t-test pour comparer deux groupes (variances inégales).
        
        Returns:
            t_statistic, p_value, effect_size (Cohen's d)
        """
        n1, n2 = len(group1), len(group2)
        m1, m2 = np.mean(group1), np.mean(group2)
        v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Welch's t-statistic
        se = np.sqrt(v1/n1 + v2/n2)
        t_stat = (m1 - m2) / se if se > 0 else 0
        
        # Degrees of freedom (Welch-Satterthwaite)
        df = ((v1/n1 + v2/n2)**2) / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
        
        # p-value (two-tailed, approximation using normal for simplicity)
        from scipy.stats import t as t_dist
        try:
            p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df))
        except:
            # Fallback if scipy not available
            p_value = 2 * (1 - 0.5 * (1 + np.tanh(t_stat / np.sqrt(2))))
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
        cohens_d = (m1 - m2) / pooled_std if pooled_std > 0 else 0
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'df': float(df),
            'significant_05': p_value < 0.05,
            'significant_01': p_value < 0.01
        }
    
    @staticmethod
    def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict:
        """
        Correction de Bonferroni pour comparaisons multiples.
        """
        n = len(p_values)
        corrected_alpha = alpha / n
        
        return {
            'original_alpha': alpha,
            'corrected_alpha': corrected_alpha,
            'n_comparisons': n,
            'significant': [p < corrected_alpha for p in p_values],
            'corrected_p_values': [min(p * n, 1.0) for p in p_values]
        }
    
    @staticmethod
    def bootstrap_ci(
        data: List[float],
        n_bootstrap: int = 10000,
        confidence: float = 0.95,
        statistic: Callable = np.mean
    ) -> Tuple[float, float]:
        """
        Intervalle de confiance par bootstrap.
        
        Plus robuste que les méthodes paramétriques pour petits échantillons.
        """
        arr = np.array(data)
        n = len(arr)
        
        # Bootstrap sampling
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(arr, size=n, replace=True)
            bootstrap_stats.append(statistic(sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Percentile method
        lower = np.percentile(bootstrap_stats, (1 - confidence) / 2 * 100)
        upper = np.percentile(bootstrap_stats, (1 + confidence) / 2 * 100)
        
        return float(lower), float(upper)


# =============================================================================
# EXPERIMENT LOGGING
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration complète d'une expérience."""
    experiment_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Environment
    env_name: str = ""
    env_params: Dict = field(default_factory=dict)
    
    # Agent
    agent_type: str = ""
    agent_params: Dict = field(default_factory=dict)
    
    # Training
    n_episodes: int = 1000
    n_seeds: int = 5
    seeds: List[int] = field(default_factory=list)
    
    # System
    python_version: str = field(default_factory=lambda: sys.version)
    numpy_version: str = field(default_factory=lambda: np.__version__)
    torch_version: str = ""
    cuda_available: bool = False
    device: str = "cpu"
    
    # Git
    git_commit: str = ""
    git_branch: str = ""
    
    def __post_init__(self):
        # Try to get PyTorch info
        try:
            import torch
            self.torch_version = torch.__version__
            self.cuda_available = torch.cuda.is_available()
            self.device = "cuda" if self.cuda_available else "cpu"
        except ImportError:
            pass
        
        # Try to get git info
        try:
            import subprocess
            self.git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()[:8]
            self.git_branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
        except:
            pass
    
    def config_hash(self) -> str:
        """Génère un hash unique pour cette configuration."""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


class ExperimentLogger:
    """
    Logger complet pour expériences reproductibles.
    """
    
    def __init__(self, results_dir: str = "results/experiments"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.experiments = {}
        
    def start_experiment(self, config: ExperimentConfig) -> str:
        """Démarre une nouvelle expérience et retourne son ID."""
        exp_id = f"{config.experiment_name}_{config.config_hash()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiments[exp_id] = {
            'config': asdict(config),
            'runs': [],
            'status': 'running',
            'start_time': datetime.now().isoformat()
        }
        
        # Save initial config
        exp_dir = self.results_dir / exp_id
        exp_dir.mkdir(exist_ok=True)
        
        with open(exp_dir / 'config.json', 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        return exp_id
    
    def log_run(
        self,
        exp_id: str,
        seed: int,
        metrics: Dict[str, float],
        episode_data: Optional[Dict] = None
    ):
        """Log un run individuel."""
        run_data = {
            'seed': seed,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        if episode_data:
            run_data['episode_data'] = episode_data
        
        self.experiments[exp_id]['runs'].append(run_data)
        
        # Save incrementally
        exp_dir = self.results_dir / exp_id
        runs_file = exp_dir / 'runs.json'
        
        with open(runs_file, 'w') as f:
            json.dump(self.experiments[exp_id]['runs'], f, indent=2)
    
    def finish_experiment(self, exp_id: str) -> Dict:
        """Finalise une expérience et génère le résumé statistique."""
        exp = self.experiments[exp_id]
        exp['status'] = 'completed'
        exp['end_time'] = datetime.now().isoformat()
        
        # Compute statistical summary
        analyzer = StatisticalAnalyzer()
        
        # Aggregate metrics across runs
        all_metrics = {}
        for metric_name in exp['runs'][0]['metrics'].keys():
            values = [run['metrics'][metric_name] for run in exp['runs']]
            all_metrics[metric_name] = asdict(analyzer.compute_summary(values))
        
        exp['summary'] = all_metrics
        
        # Save final results
        exp_dir = self.results_dir / exp_id
        with open(exp_dir / 'final_results.json', 'w') as f:
            json.dump(exp, f, indent=2)
        
        # Generate report
        self._generate_report(exp_id, exp)
        
        return exp
    
    def _generate_report(self, exp_id: str, exp: Dict):
        """Génère un rapport textuel."""
        lines = [
            f"# Experiment Report: {exp['config']['experiment_name']}",
            f"",
            f"**ID:** {exp_id}",
            f"**Status:** {exp['status']}",
            f"**Start:** {exp.get('start_time', 'N/A')}",
            f"**End:** {exp.get('end_time', 'N/A')}",
            f"",
            "## Configuration",
            f"- Agent: {exp['config']['agent_type']}",
            f"- Environment: {exp['config']['env_name']}",
            f"- Episodes: {exp['config']['n_episodes']}",
            f"- Seeds: {exp['config']['seeds']}",
            f"",
            "## Results Summary",
            ""
        ]
        
        if 'summary' in exp:
            lines.append("| Metric | Mean | Std | 95% CI |")
            lines.append("|--------|------|-----|--------|")
            
            for metric, stats in exp['summary'].items():
                ci = f"[{stats['ci_95_lower']:.2f}, {stats['ci_95_upper']:.2f}]"
                lines.append(f"| {metric} | {stats['mean']:.4f} | {stats['std']:.4f} | {ci} |")
        
        lines.extend([
            "",
            "## Reproducibility Info",
            f"- Python: {exp['config']['python_version'].split()[0]}",
            f"- NumPy: {exp['config']['numpy_version']}",
            f"- PyTorch: {exp['config'].get('torch_version', 'N/A')}",
            f"- Git commit: {exp['config'].get('git_commit', 'N/A')}",
            f"- Git branch: {exp['config'].get('git_branch', 'N/A')}",
        ])
        
        report = "\n".join(lines)
        
        exp_dir = self.results_dir / exp_id
        with open(exp_dir / 'report.md', 'w') as f:
            f.write(report)


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

class ExperimentComparator:
    """Compare plusieurs expériences avec tests statistiques."""
    
    def __init__(self, experiments: Dict[str, List[float]]):
        """
        Args:
            experiments: Dict mapping experiment name to list of metric values
        """
        self.experiments = experiments
        self.analyzer = StatisticalAnalyzer()
        
    def pairwise_comparison(self) -> Dict[str, Dict]:
        """Compare toutes les paires d'expériences."""
        names = list(self.experiments.keys())
        comparisons = {}
        
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                key = f"{name1}_vs_{name2}"
                comparisons[key] = self.analyzer.welch_t_test(
                    self.experiments[name1],
                    self.experiments[name2]
                )
        
        return comparisons
    
    def rank_experiments(self) -> List[Tuple[str, StatisticalSummary]]:
        """Classe les expériences par performance moyenne."""
        summaries = [
            (name, self.analyzer.compute_summary(values))
            for name, values in self.experiments.items()
        ]
        return sorted(summaries, key=lambda x: x[1].mean, reverse=True)
    
    def generate_comparison_table(self) -> str:
        """Génère une table de comparaison formatée."""
        rankings = self.rank_experiments()
        
        lines = [
            "| Rank | Experiment | Mean ± Std | 95% CI | Significant vs Next |",
            "|------|------------|------------|--------|---------------------|"
        ]
        
        pairwise = self.pairwise_comparison()
        
        for i, (name, stats) in enumerate(rankings):
            ci = f"[{stats.ci_95_lower:.2f}, {stats.ci_95_upper:.2f}]"
            
            # Check significance vs next
            if i < len(rankings) - 1:
                next_name = rankings[i + 1][0]
                key = f"{name}_vs_{next_name}"
                if key not in pairwise:
                    key = f"{next_name}_vs_{name}"
                sig = "Yes" if pairwise.get(key, {}).get('significant_05', False) else "No"
            else:
                sig = "-"
            
            lines.append(
                f"| {i+1} | {name} | {stats.mean:.2f} ± {stats.std:.2f} | {ci} | {sig} |"
            )
        
        return "\n".join(lines)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_reproducible_experiment(
    env_factory: Callable,
    agent_factory: Callable,
    experiment_name: str,
    n_episodes: int = 1000,
    n_seeds: int = 5,
    results_dir: str = "results/experiments"
) -> Dict:
    """
    Exécute une expérience complète avec reproductibilité.
    
    Usage:
        results = run_reproducible_experiment(
            env_factory=lambda: GridWorld(),
            agent_factory=lambda seed: DQNAgent(...),
            experiment_name="dqn_gridworld",
            n_episodes=1000,
            n_seeds=5
        )
    """
    # Setup
    seed_manager = SeedManager()
    logger = ExperimentLogger(results_dir)
    seeds = seed_manager.get_default_seeds(n_seeds)
    
    # Create config
    config = ExperimentConfig(
        experiment_name=experiment_name,
        n_episodes=n_episodes,
        n_seeds=n_seeds,
        seeds=seeds
    )
    
    exp_id = logger.start_experiment(config)
    print(f"Started experiment: {exp_id}")
    
    all_returns = []
    
    for i, seed in enumerate(seeds):
        print(f"  Run {i+1}/{n_seeds} (seed={seed})...")
        
        # Set seeds
        seed_manager.set_all_seeds(seed)
        
        # Create env and agent
        env = env_factory()
        agent = agent_factory(seed)
        
        # Train
        episode_returns = []
        for ep in range(n_episodes):
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
        
        all_returns.append(episode_returns)
        
        # Log run
        metrics = {
            'mean_return': np.mean(episode_returns),
            'final_return': np.mean(episode_returns[-100:]),
            'max_return': np.max(episode_returns)
        }
        logger.log_run(exp_id, seed, metrics, {'returns': episode_returns[-100:]})
    
    # Finish
    results = logger.finish_experiment(exp_id)
    print(f"Experiment completed: {exp_id}")
    
    return results


if __name__ == "__main__":
    # Demo
    print("Reproducibility Module Demo")
    print("=" * 50)
    
    # Test seed manager
    sm = SeedManager(42)
    print(f"Default seeds: {sm.get_default_seeds(5)}")
    
    # Test statistical analyzer
    data1 = [10, 12, 11, 13, 9, 14, 10, 11]
    data2 = [15, 16, 14, 17, 15, 18, 16, 15]
    
    analyzer = StatisticalAnalyzer()
    
    summary1 = analyzer.compute_summary(data1)
    print(f"\nData1 summary:")
    print(f"  Mean: {summary1.mean:.2f} ± {summary1.std:.2f}")
    print(f"  95% CI: [{summary1.ci_95_lower:.2f}, {summary1.ci_95_upper:.2f}]")
    
    test_result = analyzer.welch_t_test(data1, data2)
    print(f"\nWelch's t-test (Data1 vs Data2):")
    print(f"  t = {test_result['t_statistic']:.3f}, p = {test_result['p_value']:.4f}")
    print(f"  Cohen's d = {test_result['cohens_d']:.3f}")
    print(f"  Significant (α=0.05): {test_result['significant_05']}")
