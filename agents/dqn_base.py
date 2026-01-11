from typing import Dict, List, Tuple
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity: int, prioritized: bool = False, alpha: float = 0.6):
        self.capacity = capacity
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool, float]] = []
        self.pos = 0
        self.prioritized = prioritized
        self.alpha = alpha
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool, float]) -> None:
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        if self.prioritized:
            prios = self.priorities if len(self.buffer) == self.capacity else self.priorities[: self.pos]
            probs = prios ** self.alpha
            probs /= probs.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            weights = (len(self.buffer) * probs[indices]) ** (-beta)
            weights /= weights.max()
        else:
            indices = np.random.choice(len(self.buffer), batch_size)
            weights = np.ones_like(indices, dtype=np.float32)
        transitions = [self.buffer[i] for i in indices]
        batch = tuple(zip(*transitions))
        return batch, torch.tensor(weights, dtype=torch.float32), torch.tensor(indices, dtype=torch.long)

    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor) -> None:
        for idx, td in zip(indices, td_errors.detach().cpu().numpy()):
            self.priorities[idx] = abs(td) + 1e-6

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(self, obs_dim: int, action_dim: int, lr: float = 1e-3, gamma: float = 0.99, prioritized: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = QNetwork(obs_dim, action_dim).to(self.device)
        self.target = QNetwork(obs_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.memory = ReplayBuffer(100_000, prioritized=prioritized)
        self.steps = 0

    def select_action(self, state: np.ndarray, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(self.policy.net[-1].out_features)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def push(self, s, a, r, s2, done, td_error_estimate: float = 1.0) -> None:
        self.memory.push((s, a, r, s2, done, td_error_estimate))

    def update(self, batch_size: int = 64, beta: float = 0.4) -> Dict[str, float]:
        if len(self.memory) < batch_size:
            return {}
        (states, actions, rewards, next_states, dones, _), weights, indices = self.memory.sample(batch_size, beta)
        states_t = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(np.stack(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        weights_t = weights.to(self.device)

        q_values = self.policy(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_states_t).max(1)[0]
            target = rewards_t + self.gamma * next_q * (1 - dones_t)
        td_errors = target - q_values
        loss = (weights_t * td_errors.pow(2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        self.memory.update_priorities(indices, td_errors)
        if self.steps % 200 == 0:
            self.target.load_state_dict(self.policy.state_dict())
        self.steps += 1
        return {"loss": float(loss.item()), "td_mean": float(td_errors.abs().mean().item())}

    def save(self, path: str) -> None:
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state)
        self.target.load_state_dict(self.policy.state_dict())
