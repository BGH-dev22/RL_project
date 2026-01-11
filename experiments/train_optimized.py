"""
🚀 Entraînement Optimisé pour Meilleures Performances
=====================================================

Ce script utilise toutes les techniques d'optimisation connues pour 
maximiser les performances sur le GridWorld.

Améliorations:
1. Double DQN (réduit la surestimation)
2. Learning rate scheduling (decay progressif)
3. Epsilon decay plus lent (meilleure exploration)
4. Reward shaping (récompenses intermédiaires)
5. Plus d'épisodes d'entraînement
6. Réseau plus large

Usage:
    python experiments/train_optimized.py --episodes 5000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from collections import deque
import random

from env.gridworld import GridWorld


class ImprovedQNetwork(nn.Module):
    """Réseau plus large avec Dueling DQN architecture."""
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        # Feature extractor commun
        self.features = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Advantage stream (Dueling DQN)
        self.advantage = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        features = self.features(x)
        value = self.value(features)
        advantage = self.advantage(features)
        # Q = V + (A - mean(A)) - Dueling DQN formula
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


class PrioritizedReplayBuffer:
    """Buffer avec priorité basée sur TD-error."""
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        samples = [self.buffer[i] for i in indices]
        batch = list(zip(*samples))
        
        return batch, weights, indices
    
    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = abs(td) + 1e-6
    
    def __len__(self):
        return len(self.buffer)


class OptimizedDQNAgent:
    """Agent DQN optimisé avec toutes les améliorations."""
    
    def __init__(self, obs_dim: int, action_dim: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        
        # Réseaux (Dueling Double DQN)
        self.policy = ImprovedQNetwork(obs_dim, action_dim).to(self.device)
        self.target = ImprovedQNetwork(obs_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        
        # Optimizer avec weight decay
        self.optimizer = optim.Adam(self.policy.parameters(), lr=5e-4, weight_decay=1e-5)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        
        # Hyperparamètres optimisés
        self.gamma = 0.99
        self.batch_size = 64
        self.target_update_freq = 500  # Plus fréquent
        
        # Replay buffer
        self.memory = PrioritizedReplayBuffer(200_000, alpha=0.6)
        
        # Compteur
        self.steps = 0
        self.beta_start = 0.4
        self.beta_frames = 10000
    
    def select_action(self, state: np.ndarray, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy(state_t)
            return q_values.argmax().item()
    
    def push(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return None
        
        # Beta annealing pour importance sampling
        beta = min(1.0, self.beta_start + self.steps * (1.0 - self.beta_start) / self.beta_frames)
        
        batch, weights, indices = self.memory.sample(self.batch_size, beta)
        
        states = torch.FloatTensor(np.array(batch[0])).to(self.device)
        actions = torch.LongTensor(batch[1]).to(self.device)
        rewards = torch.FloatTensor(batch[2]).to(self.device)
        next_states = torch.FloatTensor(np.array(batch[3])).to(self.device)
        dones = torch.FloatTensor(batch[4]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q values
        current_q = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use policy network to select action, target network to evaluate
        with torch.no_grad():
            next_actions = self.policy(next_states).argmax(1)
            next_q = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # TD errors for prioritization
        td_errors = (current_q - target_q).detach().cpu().numpy()
        self.memory.update_priorities(indices, np.abs(td_errors))
        
        # Weighted loss (Huber loss for stability)
        loss = (weights * nn.functional.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.optimizer.step()
        
        self.steps += 1
        
        # Target network update
        if self.steps % self.target_update_freq == 0:
            self.target.load_state_dict(self.policy.state_dict())
        
        return loss.item()
    
    def step_scheduler(self):
        self.scheduler.step()


class RewardShaper:
    """Reward shaping pour guider l'exploration."""
    
    def __init__(self, env):
        self.env = env
        self.previous_distance_to_key = None
        self.previous_distance_to_door = None
        self.previous_distance_to_goal = None
    
    def reset(self):
        self.previous_distance_to_key = None
        self.previous_distance_to_door = None
        self.previous_distance_to_goal = None
    
    def shape_reward(self, obs, reward, info):
        """Ajoute des récompenses intermédiaires basées sur la progression."""
        shaped_reward = reward
        
        agent_pos = obs['agent'][:2]
        has_key = obs['agent'][2] > 0.5
        door_open = obs['agent'][3] > 0.5 if len(obs['agent']) > 3 else False
        
        key_pos = obs.get('key', None)
        door_pos = obs.get('door', None)
        goal_pos = obs.get('goal', None)
        
        # Reward shaping basé sur la distance
        if not has_key and key_pos is not None:
            # Récompense pour se rapprocher de la clé
            dist = np.linalg.norm(agent_pos - key_pos[:2])
            if self.previous_distance_to_key is not None:
                progress = self.previous_distance_to_key - dist
                shaped_reward += progress * 0.1  # Petit bonus pour progression
            self.previous_distance_to_key = dist
            
        elif has_key and not door_open and door_pos is not None:
            # Récompense pour se rapprocher de la porte
            dist = np.linalg.norm(agent_pos - door_pos[:2])
            if self.previous_distance_to_door is not None:
                progress = self.previous_distance_to_door - dist
                shaped_reward += progress * 0.1
            self.previous_distance_to_door = dist
            
        elif door_open and goal_pos is not None:
            # Récompense pour se rapprocher du goal
            dist = np.linalg.norm(agent_pos - goal_pos[:2])
            if self.previous_distance_to_goal is not None:
                progress = self.previous_distance_to_goal - dist
                shaped_reward += progress * 0.1
            self.previous_distance_to_goal = dist
        
        # Bonus pour les étapes clés
        if info.get('picked_key', False):
            shaped_reward += 5.0  # Bonus pour avoir pris la clé
        if info.get('opened_door', False):
            shaped_reward += 10.0  # Bonus pour avoir ouvert la porte
        
        return shaped_reward


def train_optimized(num_episodes: int = 5000, save_every: int = 500):
    """Entraînement optimisé."""
    
    print("=" * 60)
    print("🚀 ENTRAÎNEMENT OPTIMISÉ DQN")
    print("=" * 60)
    print(f"Épisodes: {num_episodes}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()
    
    # Environnement
    env = GridWorld(size=10, max_steps=200)
    obs = env.reset()
    state_dim = len(obs['agent'])
    action_dim = env.action_space
    
    # Agent optimisé
    agent = OptimizedDQNAgent(state_dim, action_dim)
    
    # Reward shaper
    shaper = RewardShaper(env)
    
    # Métriques
    returns = []
    keys_collected = []
    doors_opened = []
    goals_reached = []
    
    recent_returns = deque(maxlen=100)
    recent_goals = deque(maxlen=100)
    best_goal_rate = 0.0
    
    # Epsilon schedule plus lent
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = num_episodes * 0.8  # 80% des épisodes pour décroître
    
    for episode in range(num_episodes):
        obs = env.reset()
        state = obs['agent'].astype(np.float32)
        shaper.reset()
        
        episode_return = 0
        done = False
        
        # Epsilon décroissant
        eps = eps_end + (eps_start - eps_end) * np.exp(-episode / eps_decay)
        
        while not done:
            action = agent.select_action(state, eps)
            obs, reward, done, info = env.step(action)
            next_state = obs['agent'].astype(np.float32)
            
            # Reward shaping
            shaped_reward = shaper.shape_reward(obs, reward, info)
            
            agent.push(state, action, shaped_reward, next_state, done)
            agent.update()
            
            state = next_state
            episode_return += reward  # On garde le reward original pour les métriques
        
        # Mise à jour du scheduler
        agent.step_scheduler()
        
        # Métriques
        returns.append(episode_return)
        keys_collected.append(1 if info.get('has_key', False) else 0)
        doors_opened.append(1 if info.get('door_open', False) else 0)
        goals_reached.append(1 if info.get('reached_goal', False) else 0)
        
        recent_returns.append(episode_return)
        recent_goals.append(goals_reached[-1])
        
        # Affichage
        if (episode + 1) % 100 == 0:
            avg_return = np.mean(recent_returns)
            goal_rate = np.mean(recent_goals) * 100
            key_rate = np.mean(keys_collected[-100:]) * 100
            door_rate = np.mean(doors_opened[-100:]) * 100
            
            if goal_rate > best_goal_rate:
                best_goal_rate = goal_rate
                marker = " ⭐ NEW BEST!"
            else:
                marker = ""
            
            print(f"Episode {episode + 1:5d} | "
                  f"Return: {avg_return:7.1f} | "
                  f"Keys: {key_rate:5.1f}% | "
                  f"Doors: {door_rate:5.1f}% | "
                  f"Goals: {goal_rate:5.1f}%{marker} | "
                  f"ε: {eps:.3f}")
        
        # Sauvegarde
        if (episode + 1) % save_every == 0:
            save_dir = Path("results/optimized")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save(agent.policy.state_dict(), save_dir / f"model_ep{episode + 1}.pt")
    
    # Résultats finaux
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS FINAUX")
    print("=" * 60)
    
    final_100 = {
        'return': np.mean(returns[-100:]),
        'keys': np.mean(keys_collected[-100:]) * 100,
        'doors': np.mean(doors_opened[-100:]) * 100,
        'goals': np.mean(goals_reached[-100:]) * 100
    }
    
    print(f"Retour moyen (100 derniers): {final_100['return']:.1f}")
    print(f"Clés collectées: {final_100['keys']:.1f}%")
    print(f"Portes ouvertes: {final_100['doors']:.1f}%")
    print(f"Goals atteints: {final_100['goals']:.1f}%")
    print(f"Meilleur taux de goals: {best_goal_rate:.1f}%")
    
    # Sauvegarder les métriques
    save_dir = Path("results/optimized")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        'returns': returns,
        'keys_collected': keys_collected,
        'doors_opened': doors_opened,
        'goals_reached': goals_reached,
        'final_metrics': final_100,
        'best_goal_rate': best_goal_rate,
        'num_episodes': num_episodes
    }
    
    with open(save_dir / "optimized_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Résultats sauvegardés dans {save_dir}/")
    
    return final_100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000, help="Nombre d'épisodes")
    args = parser.parse_args()
    
    train_optimized(num_episodes=args.episodes)
