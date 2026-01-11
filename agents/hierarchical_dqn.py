from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .dqn_base import QNetwork, ReplayBuffer


class SubgoalManager(nn.Module):
    def __init__(self, obs_dim: int, subgoal_dim: int):
        super().__init__()
        self.net = QNetwork(obs_dim, subgoal_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HierarchicalDQN:
    def __init__(self, obs_dim: int, action_dim: int, subgoal_dim: int, gamma: float = 0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Larger network for low-level (handles more complex state+subgoal)
        self.low_level = nn.Sequential(
            nn.Linear(obs_dim + subgoal_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        ).to(self.device)
        self.low_target = nn.Sequential(
            nn.Linear(obs_dim + subgoal_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        ).to(self.device)
        self.low_target.load_state_dict(self.low_level.state_dict())
        self.high_level = SubgoalManager(obs_dim, subgoal_dim).to(self.device)
        self.high_target = SubgoalManager(obs_dim, subgoal_dim).to(self.device)
        self.high_target.load_state_dict(self.high_level.state_dict())
        # Lower learning rate for stability
        self.low_opt = optim.Adam(self.low_level.parameters(), lr=5e-4)
        self.high_opt = optim.Adam(self.high_level.parameters(), lr=1e-4)
        self.gamma = gamma
        self.low_buffer = ReplayBuffer(100_000, prioritized=True)
        self.high_buffer = ReplayBuffer(50_000, prioritized=True)
        self.steps = 0
        self.subgoal_dim = subgoal_dim
        self.action_dim = action_dim

    def select_subgoal(self, state: np.ndarray, eps: float) -> np.ndarray:
        if np.random.rand() < eps:
            vec = np.zeros(self.subgoal_dim, dtype=np.float32)
            vec[np.random.randint(0, self.subgoal_dim)] = 1.0
            return vec
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.high_level(s)
            idx = int(torch.argmax(q, dim=1).item())
            vec = np.zeros(self.subgoal_dim, dtype=np.float32)
            vec[idx] = 1.0
            return vec

    def select_action(self, state: np.ndarray, subgoal: np.ndarray, eps: float) -> int:
        if np.random.rand() < eps:
            return np.random.randint(0, self.action_dim)
        with torch.no_grad():
            s = torch.tensor(np.concatenate([state, subgoal], axis=-1), dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.low_level(s)
            return int(torch.argmax(q, dim=1).item())

    def push_low(self, s, sg, a, r, s2, done, td_error_estimate=1.0):
        self.low_buffer.push((np.concatenate([s, sg]), a, r, np.concatenate([s2, sg]), done, td_error_estimate))

    def push_high(self, s, sg, r, s2, done, td_error_estimate=1.0):
        self.high_buffer.push((s, sg.argmax(), r, s2, done, td_error_estimate))

    def update_low(self, batch_size: int = 64, beta: float = 0.4) -> Dict[str, float]:
        if len(self.low_buffer) < batch_size:
            return {}
        (states, actions, rewards, next_states, dones, _), weights, indices = self.low_buffer.sample(batch_size, beta)
        device = self.device
        states_t = torch.tensor(np.stack(states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states_t = torch.tensor(np.stack(next_states), dtype=torch.float32, device=device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device)
        weights_t = weights.to(device)

        q = self.low_level(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.low_target(next_states_t).max(1)[0]
            target = rewards_t + self.gamma * next_q * (1 - dones_t)
        td = target - q
        loss = (weights_t * td.pow(2)).mean()
        self.low_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.low_level.parameters(), 1.0)
        self.low_opt.step()
        self.low_buffer.update_priorities(indices, td)
        if self.steps % 200 == 0:
            self.low_target.load_state_dict(self.low_level.state_dict())
        return {"low_loss": float(loss.item()), "low_td": float(td.abs().mean().item())}

    def update_high(self, batch_size: int = 64, beta: float = 0.4) -> Dict[str, float]:
        if len(self.high_buffer) < batch_size:
            return {}
        (states, actions, rewards, next_states, dones, _), weights, indices = self.high_buffer.sample(batch_size, beta)
        device = self.device
        states_t = torch.tensor(np.stack(states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states_t = torch.tensor(np.stack(next_states), dtype=torch.float32, device=device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device)
        weights_t = weights.to(device)

        q = self.high_level(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.high_target(next_states_t).max(1)[0]
            target = rewards_t + self.gamma * next_q * (1 - dones_t)
        td = target - q
        loss = (weights_t * td.pow(2)).mean()
        self.high_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.high_level.parameters(), 1.0)
        self.high_opt.step()
        self.high_buffer.update_priorities(indices, td)
        if self.steps % 400 == 0:
            self.high_target.load_state_dict(self.high_level.state_dict())
        self.steps += 1
        return {"high_loss": float(loss.item()), "high_td": float(td.abs().mean().item())}
