"""
Policy Gradient Baselines: A2C, PPO, SAC
=========================================
Implémentation des algorithmes de référence pour comparaison équitable.

Références:
- A2C: Mnih et al., "Asynchronous Methods for Deep RL", 2016
- PPO: Schulman et al., "Proximal Policy Optimization", 2017
- SAC: Haarnoja et al., "Soft Actor-Critic", 2018

Auteur: ProRL Project
Date: 2025
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from collections import deque


# =============================================================================
# NETWORKS
# =============================================================================

class ActorCriticNetwork(nn.Module):
    """Réseau partagé pour A2C et PPO (actions discrètes)."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        return self.actor(features), self.critic(features)
    
    def get_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value


class GaussianActorCritic(nn.Module):
    """Réseau pour actions continues (SAC, PPO continu)."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std(features), -20, 2)
        return mean, log_std, self.critic(features)


class SACQNetwork(nn.Module):
    """Double Q-Network pour SAC."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


# =============================================================================
# A2C - ADVANTAGE ACTOR-CRITIC
# =============================================================================

class A2C:
    """
    Advantage Actor-Critic (A2C)
    
    Algorithme synchrone combinant policy gradient et value function.
    Utilise l'avantage A(s,a) = Q(s,a) - V(s) pour réduire la variance.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_steps: int = 5,
        seed: Optional[int] = None
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.network = ActorCriticNetwork(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        
        # Buffers pour n-step returns
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        self.training_info = {'policy_loss': [], 'value_loss': [], 'entropy': []}
        
    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Select action and return (action, log_prob, value) for training."""
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, log_prob, value = self.network.get_action(state_t)
        return action, log_prob, value
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                         done: bool, log_prob: torch.Tensor, value: torch.Tensor):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
    def compute_returns(self, next_value: torch.Tensor) -> torch.Tensor:
        """Compute n-step returns avec GAE simplifié."""
        returns = []
        R = next_value
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step] + self.gamma * R * (1 - self.dones[step])
            returns.insert(0, R)
        return torch.cat(returns)
    
    def update(self, next_state: np.ndarray) -> Dict[str, float]:
        if len(self.states) < self.n_steps:
            return {}
        
        # Compute bootstrap value
        with torch.no_grad():
            next_state_t = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, next_value = self.network(next_state_t)
        
        returns = self.compute_returns(next_value.squeeze())
        
        log_probs = torch.stack(self.log_probs).squeeze()
        values = torch.cat(self.values).squeeze()
        
        # Advantage = Returns - Values
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss (negative because we want to maximize)
        policy_loss = -(log_probs * advantages).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns.detach())
        
        # Entropy bonus pour l'exploration
        states_t = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        logits, _ = self.network(states_t)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Clear buffers
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        
        info = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
        for k, v in info.items():
            self.training_info[k].append(v)
            
        return info


# =============================================================================
# PPO - PROXIMAL POLICY OPTIMIZATION
# =============================================================================

class PPO:
    """
    Proximal Policy Optimization (PPO)
    
    Amélioration de TRPO avec une contrainte de clip plus simple.
    Permet des mises à jour plus grandes tout en restant stable.
    
    Ref: Schulman et al., 2017
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        rollout_length: int = 2048,
        seed: Optional[int] = None
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.network = ActorCriticNetwork(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rollout_length = rollout_length
        
        # Rollout buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        self.training_info = {'policy_loss': [], 'value_loss': [], 'entropy': [], 'clip_fraction': []}
        
    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, log_prob, value = self.network.get_action(state_t)
        return action, log_prob, value
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         done: bool, log_prob: torch.Tensor, value: torch.Tensor):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        
    def compute_gae(self, next_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        values = torch.cat(self.values + [next_value]).squeeze()
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)
        
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            
        returns = advantages + values[:-1]
        return advantages, returns
    
    def update(self, next_state: np.ndarray) -> Dict[str, float]:
        if len(self.states) < self.rollout_length:
            return {}
        
        # Compute advantages and returns
        with torch.no_grad():
            next_state_t = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, next_value = self.network(next_state_t)
        
        advantages, returns = self.compute_gae(next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states_t = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.stack(self.log_probs).squeeze()
        
        # PPO update epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_clip_fraction = 0
        n_updates = 0
        
        indices = np.arange(len(self.states))
        
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, len(self.states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_states = states_t[batch_indices]
                batch_actions = actions_t[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                logits, values = self.network(batch_states)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                clip_fraction = ((ratio - 1).abs() > self.clip_epsilon).float().mean().item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_clip_fraction += clip_fraction
                n_updates += 1
        
        # Clear buffers
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        
        info = {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'clip_fraction': total_clip_fraction / n_updates
        }
        for k, v in info.items():
            self.training_info[k].append(v)
            
        return info


# =============================================================================
# SAC - SOFT ACTOR-CRITIC (DISCRETE VERSION)
# =============================================================================

class SACDiscrete:
    """
    Soft Actor-Critic pour actions discrètes.
    
    Maximise à la fois le retour attendu et l'entropie de la politique,
    permettant une exploration robuste et une convergence stable.
    
    Ref: Haarnoja et al., 2018 (adapté pour actions discrètes)
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        buffer_size: int = 100000,
        batch_size: int = 256,
        seed: Optional[int] = None
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # Networks
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        ).to(self.device)
        
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        ).to(self.device)
        
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        ).to(self.device)
        
        self.q1_target = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        ).to(self.device)
        
        self.q2_target = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        ).to(self.device)
        
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        
        # Automatic temperature tuning
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -np.log(1.0 / action_dim) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
        
        # Replay buffer
        self.buffer = deque(maxlen=buffer_size)
        
        self.training_info = {'q_loss': [], 'actor_loss': [], 'alpha': [], 'entropy': []}
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.actor(state_t)
            probs = F.softmax(logits, dim=-1)
            
            if deterministic:
                action = probs.argmax(dim=-1).item()
            else:
                dist = Categorical(probs)
                action = dist.sample().item()
        return action
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
        
    def update(self) -> Dict[str, float]:
        if len(self.buffer) < self.batch_size:
            return {}
        
        # Sample batch
        batch = [self.buffer[i] for i in np.random.choice(len(self.buffer), self.batch_size, replace=False)]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # Update Q-functions
        with torch.no_grad():
            next_logits = self.actor(next_states_t)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = torch.log(next_probs + 1e-8)
            
            next_q1 = self.q1_target(next_states_t)
            next_q2 = self.q2_target(next_states_t)
            next_q = torch.min(next_q1, next_q2)
            
            next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(dim=-1)
            target_q = rewards_t + self.gamma * (1 - dones_t) * next_v
        
        # Q1 loss
        q1_values = self.q1(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        q1_loss = F.mse_loss(q1_values, target_q)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        # Q2 loss
        q2_values = self.q2(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        q2_loss = F.mse_loss(q2_values, target_q)
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Update actor
        logits = self.actor(states_t)
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-8)
        
        q1_values = self.q1(states_t)
        q2_values = self.q2(states_t)
        q_values = torch.min(q1_values, q2_values)
        
        actor_loss = (probs * (self.alpha * log_probs - q_values)).sum(dim=-1).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        
        # Soft update target networks
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        info = {
            'q_loss': (q1_loss.item() + q2_loss.item()) / 2,
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha,
            'entropy': entropy.item()
        }
        for k, v in info.items():
            self.training_info[k].append(v)
            
        return info


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_baseline_agent(
    name: str,
    obs_dim: int,
    action_dim: int,
    seed: Optional[int] = None,
    **kwargs
):
    """Factory pour créer les agents baseline."""
    agents = {
        'a2c': A2C,
        'ppo': PPO,
        'sac': SACDiscrete
    }
    if name.lower() not in agents:
        raise ValueError(f"Unknown agent: {name}. Available: {list(agents.keys())}")
    return agents[name.lower()](obs_dim, action_dim, seed=seed, **kwargs)
