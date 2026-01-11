"""
Warehouse Robot Training with Hierarchical DQN
==============================================
Script d'entraînement pour le problème réel de robotique d'entrepôt.

Auteur: ProRL Project
Date: 2025
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from pathlib import Path
import json
import matplotlib.pyplot as plt

from env.warehouse_robot import WarehouseEnv, WarehouseSubgoals
from agents.dqn_base import ReplayBuffer


class WarehouseQNetwork(nn.Module):
    """Réseau Q adapté pour l'environnement d'entrepôt."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WarehouseHierarchicalAgent:
    """
    Agent hiérarchique pour la navigation d'entrepôt.
    
    Architecture à deux niveaux:
    - High-level: Sélection de sous-objectifs (go_to_package, pickup, etc.)
    - Low-level: Actions primitives (up, down, left, right, pickup, drop, etc.)
    """
    
    def __init__(self, obs_dim: int, action_dim: int, num_subgoals: int = 7,
                 gamma: float = 0.99, lr: float = 1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_subgoals = num_subgoals
        self.gamma = gamma
        
        # High-level network (subgoal selection)
        self.high_policy = WarehouseQNetwork(obs_dim, num_subgoals).to(self.device)
        self.high_target = WarehouseQNetwork(obs_dim, num_subgoals).to(self.device)
        self.high_target.load_state_dict(self.high_policy.state_dict())
        self.high_opt = optim.Adam(self.high_policy.parameters(), lr=lr)
        
        # Low-level network (action selection, conditioned on subgoal)
        self.low_policy = WarehouseQNetwork(obs_dim + num_subgoals, action_dim).to(self.device)
        self.low_target = WarehouseQNetwork(obs_dim + num_subgoals, action_dim).to(self.device)
        self.low_target.load_state_dict(self.low_policy.state_dict())
        self.low_opt = optim.Adam(self.low_policy.parameters(), lr=lr)
        
        # Replay buffers
        self.high_buffer = ReplayBuffer(50000, prioritized=True)
        self.low_buffer = ReplayBuffer(100000, prioritized=True)
        
        self.steps = 0
        self.current_subgoal = None
        self.subgoal_steps = 0
    
    def select_subgoal(self, state: np.ndarray, eps: float) -> int:
        """Sélectionner un sous-objectif."""
        if np.random.random() < eps:
            return np.random.randint(0, self.num_subgoals)
        
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.high_policy(state_t)
            return int(torch.argmax(q_values, dim=1).item())
    
    def select_action(self, state: np.ndarray, subgoal: int, eps: float) -> int:
        """Sélectionner une action basée sur l'état et le sous-objectif."""
        if np.random.random() < eps:
            return np.random.randint(0, self.action_dim)
        
        # Créer le vecteur one-hot pour le sous-objectif
        subgoal_vec = np.zeros(self.num_subgoals, dtype=np.float32)
        subgoal_vec[subgoal] = 1.0
        
        # Concaténer état et sous-objectif
        combined = np.concatenate([state, subgoal_vec])
        
        with torch.no_grad():
            state_t = torch.tensor(combined, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.low_policy(state_t)
            return int(torch.argmax(q_values, dim=1).item())
    
    def push_high(self, state, subgoal, reward, next_state, done):
        """Ajouter une transition high-level."""
        self.high_buffer.push((state, subgoal, reward, next_state, done, 1.0))
    
    def push_low(self, state, subgoal, action, reward, next_state, done):
        """Ajouter une transition low-level."""
        subgoal_vec = np.zeros(self.num_subgoals, dtype=np.float32)
        subgoal_vec[subgoal] = 1.0
        combined_state = np.concatenate([state, subgoal_vec])
        combined_next = np.concatenate([next_state, subgoal_vec])
        self.low_buffer.push((combined_state, action, reward, combined_next, done, 1.0))
    
    def update_high(self, batch_size: int = 64, beta: float = 0.4) -> Dict:
        """Mettre à jour le réseau high-level."""
        if len(self.high_buffer) < batch_size:
            return {}
        
        batch, weights, indices = self.high_buffer.sample(batch_size, beta)
        states, subgoals, rewards, next_states, dones, _ = batch
        
        states_t = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)
        subgoals_t = torch.tensor(subgoals, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(np.stack(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        weights_t = weights.to(self.device)
        
        q_values = self.high_policy(states_t).gather(1, subgoals_t.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q = self.high_target(next_states_t).max(1)[0]
            target = rewards_t + self.gamma * next_q * (1 - dones_t)
        
        td_errors = target - q_values
        loss = (weights_t * td_errors.pow(2)).mean()
        
        self.high_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.high_policy.parameters(), 1.0)
        self.high_opt.step()
        
        self.high_buffer.update_priorities(indices, td_errors)
        
        return {"high_loss": loss.item()}
    
    def update_low(self, batch_size: int = 64, beta: float = 0.4) -> Dict:
        """Mettre à jour le réseau low-level."""
        if len(self.low_buffer) < batch_size:
            return {}
        
        batch, weights, indices = self.low_buffer.sample(batch_size, beta)
        states, actions, rewards, next_states, dones, _ = batch
        
        states_t = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(np.stack(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        weights_t = weights.to(self.device)
        
        q_values = self.low_policy(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q = self.low_target(next_states_t).max(1)[0]
            target = rewards_t + self.gamma * next_q * (1 - dones_t)
        
        td_errors = target - q_values
        loss = (weights_t * td_errors.pow(2)).mean()
        
        self.low_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.low_policy.parameters(), 1.0)
        self.low_opt.step()
        
        self.low_buffer.update_priorities(indices, td_errors)
        
        return {"low_loss": loss.item()}
    
    def update_targets(self):
        """Mettre à jour les réseaux cibles."""
        if self.steps % 200 == 0:
            self.low_target.load_state_dict(self.low_policy.state_dict())
        if self.steps % 500 == 0:
            self.high_target.load_state_dict(self.high_policy.state_dict())
        self.steps += 1
    
    def save(self, path: str):
        """Sauvegarder l'agent."""
        torch.save({
            'high_policy': self.high_policy.state_dict(),
            'low_policy': self.low_policy.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Charger l'agent."""
        checkpoint = torch.load(path, map_location=self.device)
        self.high_policy.load_state_dict(checkpoint['high_policy'])
        self.high_target.load_state_dict(checkpoint['high_policy'])
        self.low_policy.load_state_dict(checkpoint['low_policy'])
        self.low_target.load_state_dict(checkpoint['low_policy'])


def train_warehouse_agent(
    num_episodes: int = 2000,
    max_steps: int = 300,
    log_interval: int = 100,
    save_path: str = "results/warehouse"
):
    """Entraîner l'agent sur l'environnement d'entrepôt."""
    
    print("=" * 60)
    print("ENTRAÎNEMENT: Robot d'Entrepôt avec DQN Hiérarchique")
    print("=" * 60)
    
    # Créer l'environnement
    env = WarehouseEnv(
        width=20,
        height=15,
        num_packages=3,
        num_other_robots=2,
        max_steps=max_steps
    )
    
    # Créer l'agent
    obs = env.reset()
    state = env.get_state_vector(obs)
    
    agent = WarehouseHierarchicalAgent(
        obs_dim=len(state),
        action_dim=env.action_space,
        num_subgoals=len(WarehouseSubgoals.SUBGOALS)
    )
    
    # Métriques
    metrics = {
        'returns': [],
        'packages_delivered': [],
        'missions_complete': [],
        'battery_deaths': [],
        'episode_lengths': []
    }
    
    # Epsilon decay
    eps_start, eps_end = 1.0, 0.05
    eps_decay = (eps_start - eps_end) / (num_episodes * 0.8)
    
    best_avg_return = -np.inf
    
    for episode in range(1, num_episodes + 1):
        eps = max(eps_end, eps_start - episode * eps_decay)
        
        obs = env.reset()
        state = env.get_state_vector(obs)
        
        # Sélectionner le sous-objectif initial
        subgoal = agent.select_subgoal(state, eps)
        subgoal_start_state = state.copy()
        subgoal_reward = 0
        subgoal_steps = 0
        
        total_reward = 0
        done = False
        
        while not done:
            # Sélectionner et exécuter l'action
            action = agent.select_action(state, subgoal, eps)
            obs, reward, done, info = env.step(action)
            next_state = env.get_state_vector(obs)
            
            # Ajouter à la mémoire low-level
            agent.push_low(state, subgoal, action, reward, next_state, done)
            
            subgoal_reward += reward
            subgoal_steps += 1
            total_reward += reward
            
            # Mettre à jour low-level fréquemment
            if len(agent.low_buffer) > 64:
                agent.update_low()
            
            # Changer de sous-objectif toutes les 20 steps ou si terminé
            if subgoal_steps >= 20 or done:
                # Transition high-level
                agent.push_high(subgoal_start_state, subgoal, subgoal_reward, next_state, done)
                
                if len(agent.high_buffer) > 32:
                    agent.update_high()
                
                # Nouveau sous-objectif
                subgoal = agent.select_subgoal(next_state, eps)
                subgoal_start_state = next_state.copy()
                subgoal_reward = 0
                subgoal_steps = 0
            
            agent.update_targets()
            state = next_state
        
        # Enregistrer les métriques
        metrics['returns'].append(total_reward)
        metrics['packages_delivered'].append(env.robot.packages_delivered)
        metrics['missions_complete'].append(int(info.get('mission_complete', False)))
        metrics['battery_deaths'].append(int(info.get('battery_dead', False)))
        metrics['episode_lengths'].append(env.steps)
        
        # Logging
        if episode % log_interval == 0:
            avg_return = np.mean(metrics['returns'][-log_interval:])
            avg_delivered = np.mean(metrics['packages_delivered'][-log_interval:])
            mission_rate = np.mean(metrics['missions_complete'][-log_interval:])
            battery_rate = np.mean(metrics['battery_deaths'][-log_interval:])
            
            print(f"Episode {episode:4d} | Return: {avg_return:7.1f} | " +
                  f"Delivered: {avg_delivered:.2f}/3 | " +
                  f"Complete: {mission_rate:.1%} | " +
                  f"Battery Death: {battery_rate:.1%} | eps: {eps:.3f}")
            
            if avg_return > best_avg_return:
                best_avg_return = avg_return
                agent.save(f"{save_path}/best_agent.pt")
    
    # Sauvegarder les résultats
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    with open(f"{save_path}/metrics.json", 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in metrics.items()}, f)
    
    # Générer les graphiques
    plot_warehouse_results(metrics, save_path)
    
    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT TERMINÉ")
    print("=" * 60)
    print(f"Meilleur retour moyen: {best_avg_return:.1f}")
    print(f"Résultats sauvegardés dans: {save_path}")
    
    return agent, metrics


def plot_warehouse_results(metrics: Dict, save_path: str):
    """Générer les graphiques des résultats."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Lissage
    def smooth(data, window=50):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Returns
    ax = axes[0, 0]
    ax.plot(smooth(metrics['returns']), color='blue', alpha=0.8)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.set_title('Retour par Épisode')
    ax.grid(True, alpha=0.3)
    
    # Packages delivered
    ax = axes[0, 1]
    ax.plot(smooth(metrics['packages_delivered']), color='green', alpha=0.8)
    ax.axhline(y=3, color='red', linestyle='--', alpha=0.5, label='Max')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Colis Livrés')
    ax.set_title('Colis Livrés par Épisode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mission complete rate
    ax = axes[1, 0]
    window = 100
    complete_rate = [np.mean(metrics['missions_complete'][max(0, i-window):i+1]) 
                     for i in range(len(metrics['missions_complete']))]
    ax.plot(complete_rate, color='purple', alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Taux de Mission Complète')
    ax.set_title('Taux de Succès (Mission Complète)')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Episode length
    ax = axes[1, 1]
    ax.plot(smooth(metrics['episode_lengths']), color='orange', alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Longueur Episode')
    ax.set_title('Longueur des Épisodes')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/training_plots.png", dpi=150)
    plt.close()
    
    print(f"Graphiques sauvegardés: {save_path}/training_plots.png")


def demo_warehouse(agent: WarehouseHierarchicalAgent = None, num_episodes: int = 3):
    """Démonstration de l'agent dans l'entrepôt."""
    print("\n" + "=" * 60)
    print("DÉMONSTRATION: Robot d'Entrepôt")
    print("=" * 60)
    
    env = WarehouseEnv(
        width=20,
        height=15,
        num_packages=3,
        num_other_robots=1,
        max_steps=200
    )
    
    for ep in range(num_episodes):
        print(f"\n--- Épisode {ep + 1} ---")
        obs = env.reset()
        state = env.get_state_vector(obs)
        
        if agent:
            subgoal = agent.select_subgoal(state, eps=0.05)
        
        done = False
        step = 0
        
        # Afficher état initial
        print(env.render_text())
        
        while not done and step < 50:  # Limiter pour la démo
            if agent:
                action = agent.select_action(state, subgoal, eps=0.05)
                if step % 20 == 0:
                    subgoal = agent.select_subgoal(state, eps=0.05)
            else:
                # Agent aléatoire
                action = np.random.randint(0, env.action_space)
            
            obs, reward, done, info = env.step(action)
            state = env.get_state_vector(obs)
            step += 1
            
            # Afficher tous les 10 steps
            if step % 10 == 0 or done:
                print(f"\nStep {step}:")
                print(env.render_text())
                if info:
                    print(f"Info: {info}")
        
        print(f"\nÉpisode terminé - Colis livrés: {env.robot.packages_delivered}/3")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()
    
    if args.demo:
        demo_warehouse(num_episodes=2)
    else:
        agent, metrics = train_warehouse_agent(
            num_episodes=args.episodes,
            log_interval=50,
            save_path="results/warehouse"
        )
        demo_warehouse(agent, num_episodes=2)
