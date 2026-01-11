"""
Adaptive Episodic Memory with Contextual Similarity (AEM-CS)
============================================================
INNOVATION: Une amélioration de la mémoire épisodique standard qui utilise:
1. Similarité contextuelle (pas seulement spatiale)
2. Clustering adaptatif des épisodes
3. Reconstruction de trajectoires optimales
4. Meta-learning pour ajuster les poids de priorité

Auteur: ProRL Project
Date: 2025
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    """Encode episode context into latent space for similarity computation."""
    
    def __init__(self, state_dim: int, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + 3, 64),  # +3 for has_key, near_door, near_goal
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.latent_dim = latent_dim
        
    def forward(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, context], dim=-1)
        return self.encoder(x)


class AdaptiveEpisode:
    """Enhanced episode with contextual information and abstract representation."""
    
    def __init__(
        self,
        trajectory: List[Tuple[np.ndarray, int, float]],
        return_total: float,
        td_errors: List[float],
        subgoals_achieved: List[str],
        timestamp: int,
        context_sequence: List[np.ndarray]
    ):
        self.trajectory = trajectory
        self.return_total = return_total
        self.td_errors = td_errors
        self.subgoals_achieved = subgoals_achieved  # ['key', 'door', 'goal']
        self.length = len(trajectory)
        self.timestamp = timestamp
        self.context_sequence = context_sequence
        
        # Compute abstract representation
        self.abstract_repr = self._compute_abstract_repr()
        self.rarity_scores = self._compute_rarity()
        self.transition_quality = self._compute_transition_quality()
        
    def _compute_abstract_repr(self) -> Dict:
        """Create abstract representation of episode structure."""
        repr_dict = {
            'achieved_key': 'key' in self.subgoals_achieved,
            'achieved_door': 'door' in self.subgoals_achieved,
            'achieved_goal': 'goal' in self.subgoals_achieved,
            'efficiency': self.return_total / max(1, self.length),
            'mean_td': np.mean(self.td_errors) if self.td_errors else 0.0,
            'trajectory_compactness': self._compute_compactness()
        }
        return repr_dict
    
    def _compute_compactness(self) -> float:
        """Measure how compact/efficient the trajectory is (less backtracking)."""
        if len(self.trajectory) < 2:
            return 1.0
        positions = [s[:2] for s, _, _ in self.trajectory]
        unique_positions = len(set(tuple(p) for p in positions))
        return unique_positions / len(positions)
    
    def _compute_rarity(self) -> List[float]:
        counts: Dict[Tuple[int, int], int] = {}
        for (s, _, _) in self.trajectory:
            key = tuple(s[:2].astype(int).tolist())
            counts[key] = counts.get(key, 0) + 1
        return [1.0 / counts[tuple(s[:2].astype(int).tolist())] for (s, _, _) in self.trajectory]
    
    def _compute_transition_quality(self) -> List[float]:
        """Compute quality score for each transition based on reward progression."""
        if len(self.trajectory) < 2:
            return [1.0]
        qualities = []
        for i, (s, a, r) in enumerate(self.trajectory):
            # Quality based on: positive reward, TD error (surprise), position in trajectory
            position_weight = 1.0 + (i / len(self.trajectory))  # Later transitions weighted higher
            td_weight = self.td_errors[i] if i < len(self.td_errors) else 0.0
            quality = (r + 1) * position_weight + 0.1 * abs(td_weight)
            qualities.append(max(0.1, quality))
        return qualities


class AdaptiveEpisodicMemory:
    """
    INNOVATION: Mémoire épisodique adaptative avec:
    1. Clustering automatique des épisodes par contexte
    2. Reconstruction de trajectoires optimales
    3. Meta-learning pour ajuster dynamiquement les paramètres
    """
    
    def __init__(self, capacity: int = 1000, state_dim: int = 3, device: str = 'cpu'):
        self.capacity = capacity
        self.episodes: List[AdaptiveEpisode] = []
        self.time = 0
        self.device = torch.device(device)
        
        # Context encoder for similarity computation
        self.context_encoder = ContextEncoder(state_dim, latent_dim=32).to(self.device)
        
        # Clusters for episode organization
        self.clusters: Dict[str, List[int]] = defaultdict(list)
        
        # Meta-learning parameters (adaptively tuned)
        self.meta_params = {
            'alpha': 0.4,  # Return weight
            'beta': 0.3,   # Rarity weight
            'gamma': 0.3,  # TD error weight
            'recency_decay': 0.95
        }
        
        # Statistics for meta-learning
        self.retrieval_stats = {
            'successes': [],
            'failures': [],
            'param_history': []
        }
        
    def compute_context(self, state: np.ndarray, env_info: Dict) -> np.ndarray:
        """Compute context vector from state and environment info."""
        has_key = float(state[2]) if len(state) > 2 else 0.0
        near_door = float(env_info.get('near_door', False))
        near_goal = float(env_info.get('near_goal', False))
        return np.array([has_key, near_door, near_goal], dtype=np.float32)
    
    def should_store(self, ret: float, td_max: float, subgoals: List[str], 
                     returns: List[float]) -> bool:
        """Adaptive storage decision based on episode quality."""
        if not returns:
            return True
            
        # Store if achieved new subgoal combination
        existing_patterns = set()
        for ep in self.episodes:
            existing_patterns.add(tuple(ep.subgoals_achieved))
        if tuple(subgoals) not in existing_patterns:
            return True
            
        # Store high-return episodes
        p90 = np.percentile(returns, 90)
        if ret > p90:
            return True
            
        # Store surprising episodes (high TD)
        if td_max > 5.0:
            return True
            
        # Store diverse episodes (sample diversity)
        if len(self.episodes) < self.capacity * 0.5:
            return np.random.random() < 0.3
            
        return False
    
    def add(self, episode: AdaptiveEpisode) -> None:
        """Add episode with automatic clustering."""
        self.time += 1
        episode.timestamp = self.time
        
        # Assign to cluster based on subgoals achieved
        cluster_key = '_'.join(sorted(episode.subgoals_achieved)) or 'none'
        
        self.episodes.append(episode)
        self.clusters[cluster_key].append(len(self.episodes) - 1)
        
        if len(self.episodes) > self.capacity:
            self._smart_forget()
    
    def _smart_forget(self) -> None:
        """Intelligent forgetting that maintains cluster diversity."""
        # Compute forgetting scores
        scores = []
        for i, ep in enumerate(self.episodes):
            # Find cluster
            cluster_key = '_'.join(sorted(ep.subgoals_achieved)) or 'none'
            cluster_size = len(self.clusters[cluster_key])
            
            # Favor forgetting from large clusters
            cluster_penalty = cluster_size / len(self.episodes)
            
            # Standard forgetting factors
            freq = 1.0 / (1 + np.mean(ep.rarity_scores))
            obsolescence = (self.time - ep.timestamp) / max(1, self.time)
            utility = np.mean(ep.td_errors) if ep.td_errors else 0.0
            
            # Combined score (higher = more likely to forget)
            score = (0.3 * freq + 
                    0.3 * obsolescence - 
                    0.2 * utility + 
                    0.2 * cluster_penalty)
            scores.append(score)
        
        # Remove episode with highest forgetting score
        idx = int(np.argmax(scores))
        
        # Update cluster
        removed_ep = self.episodes[idx]
        cluster_key = '_'.join(sorted(removed_ep.subgoals_achieved)) or 'none'
        if idx in self.clusters[cluster_key]:
            self.clusters[cluster_key].remove(idx)
        
        self.episodes.pop(idx)
        
        # Reindex clusters
        self._reindex_clusters()
    
    def _reindex_clusters(self) -> None:
        """Rebuild cluster indices after removal."""
        self.clusters = defaultdict(list)
        for i, ep in enumerate(self.episodes):
            cluster_key = '_'.join(sorted(ep.subgoals_achieved)) or 'none'
            self.clusters[cluster_key].append(i)
    
    def contextual_similarity(self, state: np.ndarray, context: np.ndarray, 
                              episode: AdaptiveEpisode) -> float:
        """Compute contextual similarity between current state and episode."""
        # Spatial similarity
        min_dist = min(np.linalg.norm(state[:2] - s[:2]) 
                      for (s, _, _) in episode.trajectory)
        spatial_sim = 1.0 / (1.0 + min_dist)
        
        # Context similarity (has_key, near_door, near_goal)
        ep_contexts = episode.context_sequence
        if ep_contexts:
            context_dists = [np.linalg.norm(context - c) for c in ep_contexts]
            context_sim = 1.0 / (1.0 + min(context_dists))
        else:
            context_sim = 0.5
        
        # Abstract representation similarity
        abstract_sim = 0.0
        if episode.abstract_repr['achieved_goal']:
            abstract_sim += 0.5
        if episode.abstract_repr['efficiency'] > 0:
            abstract_sim += 0.3
        if episode.abstract_repr['trajectory_compactness'] > 0.5:
            abstract_sim += 0.2
            
        return 0.4 * spatial_sim + 0.3 * context_sim + 0.3 * abstract_sim
    
    def retrieve_similar(self, state: np.ndarray, context: np.ndarray, 
                        k: int = 5, prefer_successful: bool = True) -> List[AdaptiveEpisode]:
        """Retrieve k most similar episodes with contextual matching."""
        if not self.episodes:
            return []
        
        similarities = []
        for i, ep in enumerate(self.episodes):
            sim = self.contextual_similarity(state, context, ep)
            
            # Boost successful episodes
            if prefer_successful and ep.abstract_repr['achieved_goal']:
                sim *= 1.5
                
            similarities.append((i, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [self.episodes[i] for i, _ in similarities[:k]]
    
    def reconstruct_optimal_trajectory(self, current_state: np.ndarray, 
                                       goal_state: np.ndarray) -> List[Tuple[np.ndarray, int]]:
        """
        INNOVATION: Reconstruct optimal trajectory by combining segments 
        from successful episodes.
        """
        if not self.episodes:
            return []
        
        # Find episodes that reached the goal
        successful_eps = [ep for ep in self.episodes 
                         if ep.abstract_repr['achieved_goal']]
        
        if not successful_eps:
            return []
        
        # Find best matching segments
        best_trajectory = []
        current_pos = current_state[:2].copy()
        
        for _ in range(50):  # Max 50 steps
            best_action = None
            best_progress = -np.inf
            
            for ep in successful_eps:
                for i, (s, a, r) in enumerate(ep.trajectory):
                    if np.linalg.norm(s[:2] - current_pos) < 1.5:
                        # Check if this transition makes progress
                        if i + 1 < len(ep.trajectory):
                            next_s = ep.trajectory[i + 1][0]
                            progress = (np.linalg.norm(current_pos - goal_state[:2]) - 
                                       np.linalg.norm(next_s[:2] - goal_state[:2]))
                            
                            # Weight by transition quality
                            quality = ep.transition_quality[i] if i < len(ep.transition_quality) else 1.0
                            score = progress * quality
                            
                            if score > best_progress:
                                best_progress = score
                                best_action = a
                                current_pos = next_s[:2].copy()
            
            if best_action is None:
                break
                
            best_trajectory.append((current_pos.copy(), best_action))
            
            if np.linalg.norm(current_pos - goal_state[:2]) < 1.0:
                break
        
        return best_trajectory
    
    def update_meta_params(self, retrieval_useful: bool, episode_return: float) -> None:
        """
        INNOVATION: Meta-learning to adapt memory parameters based on performance.
        """
        self.retrieval_stats['successes' if retrieval_useful else 'failures'].append(episode_return)
        
        # Every 50 retrievals, adjust parameters
        total_retrievals = len(self.retrieval_stats['successes']) + len(self.retrieval_stats['failures'])
        
        if total_retrievals > 0 and total_retrievals % 50 == 0:
            success_rate = len(self.retrieval_stats['successes']) / total_retrievals
            
            # Adjust alpha (return weight) based on success rate
            if success_rate > 0.7:
                self.meta_params['alpha'] = min(0.6, self.meta_params['alpha'] + 0.05)
            elif success_rate < 0.3:
                self.meta_params['alpha'] = max(0.2, self.meta_params['alpha'] - 0.05)
            
            # Adjust beta (rarity weight) inversely
            self.meta_params['beta'] = max(0.1, 1.0 - self.meta_params['alpha'] - self.meta_params['gamma'])
            
            self.retrieval_stats['param_history'].append(self.meta_params.copy())
    
    def get_priority_samples(self, n: int) -> List[Tuple[np.ndarray, int, float]]:
        """Get high-priority transition samples for replay."""
        if not self.episodes:
            return []
        
        all_transitions = []
        for ep in self.episodes:
            for i, (s, a, r) in enumerate(ep.trajectory):
                priority = (self.meta_params['alpha'] * (r + 50) / 100 +  # Normalized reward
                           self.meta_params['beta'] * ep.rarity_scores[i] +
                           self.meta_params['gamma'] * (ep.td_errors[i] if i < len(ep.td_errors) else 0))
                all_transitions.append((s, a, r, priority))
        
        # Sort by priority and return top n
        all_transitions.sort(key=lambda x: x[3], reverse=True)
        return [(s, a, r) for s, a, r, _ in all_transitions[:n]]
    
    def get_cluster_stats(self) -> Dict[str, Dict]:
        """Get statistics about episode clusters."""
        stats = {}
        for key, indices in self.clusters.items():
            if indices:
                eps = [self.episodes[i] for i in indices if i < len(self.episodes)]
                stats[key] = {
                    'count': len(eps),
                    'avg_return': np.mean([ep.return_total for ep in eps]),
                    'avg_length': np.mean([ep.length for ep in eps]),
                    'success_rate': np.mean([ep.abstract_repr['achieved_goal'] for ep in eps])
                }
        return stats


class EpisodicMemoryGuidedExploration:
    """
    INNOVATION: Use episodic memory to guide exploration toward 
    underexplored but promising areas.
    """
    
    def __init__(self, memory: AdaptiveEpisodicMemory, exploration_bonus: float = 0.5):
        self.memory = memory
        self.exploration_bonus = exploration_bonus
        self.visit_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        
    def compute_exploration_bonus(self, state: np.ndarray) -> float:
        """Compute exploration bonus based on episodic memory."""
        pos = tuple(state[:2].astype(int).tolist())
        self.visit_counts[pos] += 1
        
        # Basic count-based bonus
        count_bonus = 1.0 / np.sqrt(self.visit_counts[pos])
        
        # Memory-based bonus: reward visiting states from successful episodes
        memory_bonus = 0.0
        for ep in self.memory.episodes:
            if ep.abstract_repr['achieved_goal']:
                for s, _, r in ep.trajectory:
                    if np.linalg.norm(s[:2] - state[:2]) < 1.5:
                        memory_bonus += 0.1 * (r + 1)
        
        return self.exploration_bonus * (count_bonus + 0.5 * memory_bonus)
    
    def suggest_action(self, state: np.ndarray, context: np.ndarray) -> Optional[int]:
        """Suggest action based on successful episode patterns."""
        similar_eps = self.memory.retrieve_similar(state, context, k=3, prefer_successful=True)
        
        if not similar_eps:
            return None
        
        # Find most common action in similar situations
        action_counts = defaultdict(float)
        for ep in similar_eps:
            for s, a, r in ep.trajectory:
                if np.linalg.norm(s[:2] - state[:2]) < 2.0:
                    # Weight by reward
                    action_counts[a] += max(0, r + 1)
        
        if action_counts:
            return max(action_counts.keys(), key=lambda x: action_counts[x])
        return None
