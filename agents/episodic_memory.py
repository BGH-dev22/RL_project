from typing import Dict, List, Tuple
import numpy as np


class Episode:
    def __init__(self, trajectory: List[Tuple[np.ndarray, int, float]], return_total: float, td_errors: List[float], subgoals: List[np.ndarray], timestamp: int):
        self.trajectory = trajectory
        self.return_total = return_total
        self.td_errors = td_errors
        self.subgoals = subgoals
        self.length = len(trajectory)
        self.timestamp = timestamp
        self.rarity_scores = self._compute_rarity()

    def _compute_rarity(self) -> List[float]:
        counts: Dict[Tuple[int, int], int] = {}
        for (s, _, _) in self.trajectory:
            key = tuple(s[:2].tolist())
            counts[key] = counts.get(key, 0) + 1
        return [1.0 / counts[tuple(s[:2].tolist())] for (s, _, _) in self.trajectory]


class EpisodicMemory:
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.episodes: List[Episode] = []
        self.time = 0

    def should_store(self, ret: float, td_max: float, num_rare: int, returns: List[float]) -> bool:
        if not returns:
            return True
        p90 = np.percentile(returns, 90)
        if ret > p90:
            return True
        if td_max > 5.0:
            return True
        if num_rare > 3:
            return True
        return False

    def add(self, episode: Episode) -> None:
        self.time += 1
        self.episodes.append(episode)
        if len(self.episodes) > self.capacity:
            self.forget()

    def forget(self) -> None:
        # Compute forgetting priority: favor keeping recent, useful, and rare episodes.
        scores = []
        for ep in self.episodes:
            freq = 1.0 / (1 + np.mean(ep.rarity_scores))
            obsolescence = (self.time - ep.timestamp) / max(1, self.time)
            utility = np.mean(ep.td_errors) if ep.td_errors else 0.0
            score = 0.5 * freq + 0.4 * obsolescence - 0.3 * utility
            scores.append(score)
        idx = int(np.argmax(scores))
        self.episodes.pop(idx)

    def prioritize(self, alpha: float, beta: float, gamma: float) -> List[Tuple[int, float]]:
        scores = []
        rets = np.array([ep.return_total for ep in self.episodes])
        rarities = np.array([np.mean(ep.rarity_scores) for ep in self.episodes])
        uncert = np.array([np.mean(ep.td_errors) for ep in self.episodes])
        if len(self.episodes) == 0:
            return []
        def norm(x):
            return (x - x.mean()) / (x.std() + 1e-6)
        for idx, ep in enumerate(self.episodes):
            score = alpha * norm(rets)[idx] + beta * norm(rarities)[idx] + gamma * norm(uncert)[idx]
            scores.append((idx, float(score)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def similar_episodes(self, state: np.ndarray, k: int = 5) -> List[Episode]:
        if not self.episodes:
            return []
        dists = []
        for ep in self.episodes:
            d = min(np.linalg.norm(state[:2] - s[:2]) for (s, _, _) in ep.trajectory)
            dists.append(d)
        idxs = np.argsort(dists)[:k]
        return [self.episodes[int(i)] for i in idxs]
