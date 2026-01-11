"""
Visualisation Complète pour Présentation
=========================================
Script pour générer toutes les visualisations du projet ProRL.

Usage:
    python visualize_all.py              # Génère tous les graphiques
    python visualize_all.py --live       # Démonstration live du robot
    python visualize_all.py --compare    # Comparaison animée des variantes

Auteur: ProRL Project
Date: 2025
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import json
from pathlib import Path
import time

# Imports du projet
from env.gridworld import GridWorld
from env.warehouse_robot import WarehouseEnv, CellType
from agents.dqn_base import DQNAgent
from agents.hierarchical_dqn import HierarchicalDQN


def plot_learning_curves():
    """
    📊 VISUALISATION 1: Courbes d'Apprentissage Comparatives
    Montre l'évolution du retour pour chaque variante.
    """
    print("\n📊 Génération des courbes d'apprentissage...")
    
    results_dir = Path("results")
    variants = ['vanilla', 'per', 'memory', 'hier', 'full', 'full_explain']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparaison des Variantes DQN - Courbes d\'Apprentissage', fontsize=14, fontweight='bold')
    
    # Charger les données
    all_returns = {}
    for variant in variants:
        metrics_file = results_dir / f"{variant}_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                all_returns[variant] = data.get('returns', [])
    
    # Fonction de lissage
    def smooth(data, window=100):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Plot 1: Retours lissés
    ax = axes[0, 0]
    for i, variant in enumerate(variants):
        if variant in all_returns and len(all_returns[variant]) > 0:
            smoothed = smooth(all_returns[variant])
            ax.plot(smoothed, color=colors[i], label=variant, linewidth=2)
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Retour (lissé)')
    ax.set_title('Évolution du Retour par Épisode')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 2: Boxplot des retours finaux
    ax = axes[0, 1]
    final_returns = []
    labels = []
    for variant in variants:
        if variant in all_returns and len(all_returns[variant]) > 100:
            final_returns.append(all_returns[variant][-500:])
            labels.append(variant)
    if final_returns:
        bp = ax.boxplot(final_returns, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax.set_ylabel('Retour')
    ax.set_title('Distribution des Retours (500 derniers épisodes)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Taux de succès
    ax = axes[1, 0]
    summary_file = results_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        variant_names = [s['Variante'] for s in summary]
        goals = [float(s['Goals (%)']) for s in summary]
        
        bars = ax.bar(variant_names, goals, color=colors[:len(variant_names)])
        ax.set_ylabel('Taux de Succès (%)')
        ax.set_title('Taux d\'Atteinte du Goal par Variante')
        ax.set_ylim(0, 100)
        
        # Ajouter les valeurs sur les barres
        for bar, val in zip(bars, goals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Métriques multiples
    ax = axes[1, 1]
    if summary_file.exists():
        metrics = ['Clés (%)', 'Portes (%)', 'Goals (%)']
        x = np.arange(len(variant_names))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [float(s[metric]) for s in summary]
            ax.bar(x + i*width, values, width, label=metric)
        
        ax.set_ylabel('Pourcentage (%)')
        ax.set_title('Progression par Sous-objectif')
        ax.set_xticks(x + width)
        ax.set_xticklabels(variant_names)
        ax.legend()
        ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/plots/learning_curves_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Graphique sauvegardé: results/plots/learning_curves_comparison.png")


def plot_theoretical_analysis():
    """
    📐 VISUALISATION 2: Analyse Théorique des Synergies
    Montre les synergies et les bornes de convergence.
    """
    print("\n📐 Génération de l'analyse théorique...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Analyse Théorique des Variantes DQN', fontsize=14, fontweight='bold')
    
    # Données théoriques
    variants = ['vanilla', 'per', 'memory', 'hier', 'full']
    convergence_rate = [1.0, 1.25, 1.33, 1.43, 4.55]
    variance_reduction = [0, 15, 20, 25, 60]
    sample_complexity = [1.0, 0.8, 0.75, 0.7, 0.22]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    
    # Plot 1: Taux de convergence
    ax = axes[0]
    bars = ax.bar(variants, convergence_rate, color=colors)
    ax.set_ylabel('Taux de Convergence Relatif')
    ax.set_title('Vitesse de Convergence')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, convergence_rate):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{val:.2f}x', ha='center', va='bottom', fontweight='bold')
    ax.set_ylim(0, 5.5)
    
    # Plot 2: Réduction de variance
    ax = axes[1]
    bars = ax.bar(variants, variance_reduction, color=colors)
    ax.set_ylabel('Réduction de Variance (%)')
    ax.set_title('Stabilité de l\'Apprentissage')
    for bar, val in zip(bars, variance_reduction):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val}%', ha='center', va='bottom', fontweight='bold')
    ax.set_ylim(0, 75)
    
    # Plot 3: Matrice de synergie
    ax = axes[2]
    synergy_matrix = np.array([
        [0, 0.10, 0.10, 0.05],  # PER
        [0.10, 0, 0.20, 0.15],  # Memory
        [0.10, 0.20, 0, 0.15],  # Hier
        [0.05, 0.15, 0.15, 0]   # Explain
    ])
    components = ['PER', 'Memory', 'Hier', 'Explain']
    
    im = ax.imshow(synergy_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(components)))
    ax.set_yticks(range(len(components)))
    ax.set_xticklabels(components)
    ax.set_yticklabels(components)
    ax.set_title('Matrice de Synergies')
    
    # Ajouter les valeurs
    for i in range(len(components)):
        for j in range(len(components)):
            val = synergy_matrix[i, j]
            color = 'white' if val > 0.1 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Score de Synergie')
    
    plt.tight_layout()
    plt.savefig('results/plots/theoretical_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Graphique sauvegardé: results/plots/theoretical_analysis.png")


def plot_warehouse_results():
    """
    🤖 VISUALISATION 3: Résultats du Robot d'Entrepôt
    Montre la progression de l'apprentissage du robot.
    """
    print("\n🤖 Génération des résultats du robot d'entrepôt...")
    
    metrics_file = Path("results/warehouse/metrics.json")
    
    if not metrics_file.exists():
        print("⚠️ Pas de données warehouse. Exécutez d'abord train_warehouse.py")
        return
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('🤖 Robot d\'Entrepôt - Progression de l\'Apprentissage', fontsize=14, fontweight='bold')
    
    def smooth(data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Plot 1: Retours
    ax = axes[0, 0]
    returns = metrics['returns']
    ax.plot(returns, alpha=0.3, color='blue')
    ax.plot(smooth(returns), color='blue', linewidth=2, label='Lissé')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(range(len(smooth(returns))), smooth(returns), 0, 
                    where=[r > 0 for r in smooth(returns)], alpha=0.3, color='green')
    ax.fill_between(range(len(smooth(returns))), smooth(returns), 0,
                    where=[r <= 0 for r in smooth(returns)], alpha=0.3, color='red')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Retour')
    ax.set_title('Évolution du Retour')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Colis livrés
    ax = axes[0, 1]
    delivered = metrics['packages_delivered']
    ax.plot(delivered, alpha=0.3, color='green')
    ax.plot(smooth(delivered), color='green', linewidth=2)
    ax.axhline(y=3, color='red', linestyle='--', label='Maximum (3)')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Colis Livrés')
    ax.set_title('Nombre de Colis Livrés par Épisode')
    ax.legend()
    ax.set_ylim(-0.1, 3.5)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Taux de mission complète
    ax = axes[1, 0]
    window = 100
    complete = metrics['missions_complete']
    complete_rate = [np.mean(complete[max(0, i-window):i+1]) * 100 
                     for i in range(len(complete))]
    ax.plot(complete_rate, color='purple', linewidth=2)
    ax.fill_between(range(len(complete_rate)), complete_rate, alpha=0.3, color='purple')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Taux de Succès (%)')
    ax.set_title('Taux de Missions Complètes (fenêtre de 100)')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Mort batterie
    ax = axes[1, 1]
    battery_deaths = metrics['battery_deaths']
    death_rate = [np.mean(battery_deaths[max(0, i-window):i+1]) * 100 
                  for i in range(len(battery_deaths))]
    ax.plot(death_rate, color='red', linewidth=2)
    ax.fill_between(range(len(death_rate)), death_rate, alpha=0.3, color='red')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Taux de Mort Batterie (%)')
    ax.set_title('Taux de Mort par Batterie Épuisée')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/warehouse_progress.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Graphique sauvegardé: results/plots/warehouse_progress.png")


def visualize_gridworld_trajectory():
    """
    🗺️ VISUALISATION 4: Trajectoire dans le GridWorld
    Montre le chemin pris par l'agent.
    """
    print("\n🗺️ Génération de la trajectoire GridWorld...")
    
    env = GridWorld(size=10, max_steps=200)
    
    # Créer et entraîner un agent rapidement
    obs = env.reset()
    state = obs['agent'].astype(np.float32)
    
    agent = DQNAgent(obs_dim=len(state), action_dim=env.action_space, prioritized=True)
    
    # Entraînement rapide
    print("  Entraînement rapide (200 épisodes)...")
    for ep in range(200):
        obs = env.reset()
        state = obs['agent'].astype(np.float32)
        done = False
        eps = max(0.1, 1.0 - ep/150)
        
        while not done:
            action = agent.select_action(state, eps)
            obs, reward, done, info = env.step(action)
            next_state = obs['agent'].astype(np.float32)
            agent.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
    
    # Collecter une trajectoire
    obs = env.reset()
    state = obs['agent'].astype(np.float32)
    trajectory = [(state[0], state[1])]
    done = False
    
    while not done:
        action = agent.select_action(state, eps=0.05)
        obs, reward, done, info = env.step(action)
        state = obs['agent'].astype(np.float32)
        trajectory.append((state[0], state[1]))
    
    # Visualiser
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Dessiner la grille
    grid = env.grid
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'red', 'yellow', 'brown', 'green'])
    ax.imshow(grid, cmap=cmap, origin='upper')
    
    # Dessiner la trajectoire
    traj_y = [t[0] for t in trajectory]
    traj_x = [t[1] for t in trajectory]
    ax.plot(traj_x, traj_y, 'b-', linewidth=2, alpha=0.7, label='Trajectoire')
    ax.scatter(traj_x[0], traj_y[0], c='blue', s=200, marker='o', label='Départ', zorder=5)
    ax.scatter(traj_x[-1], traj_y[-1], c='red', s=200, marker='*', label='Arrivée', zorder=5)
    
    # Légende
    legend_elements = [
        mpatches.Patch(color='white', label='Sol'),
        mpatches.Patch(color='black', label='Mur'),
        mpatches.Patch(color='red', label='Piège'),
        mpatches.Patch(color='yellow', label='Clé'),
        mpatches.Patch(color='brown', label='Porte'),
        mpatches.Patch(color='green', label='Goal'),
    ]
    ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0], 
              loc='upper left', bbox_to_anchor=(1, 1))
    
    ax.set_title(f'Trajectoire de l\'Agent ({len(trajectory)} steps)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/gridworld_trajectory.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Graphique sauvegardé: results/plots/gridworld_trajectory.png")


def visualize_warehouse_live():
    """
    🎬 VISUALISATION 5: Démonstration Live du Robot
    Animation du robot dans l'entrepôt.
    """
    print("\n🎬 Démonstration live du robot d'entrepôt...")
    print("   (Fermez la fenêtre pour continuer)")
    
    env = WarehouseEnv(width=20, height=15, num_packages=3, max_steps=100)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Symboles pour l'affichage
    symbols = {
        CellType.FLOOR: '.',
        CellType.WALL: '#',
        CellType.SHELF: '▓',
        CellType.PICKUP: 'P',
        CellType.DROPOFF: 'D',
        CellType.CHARGER: '⚡',
        CellType.ROBOT: '🤖',
        CellType.PACKAGE: '□',
        CellType.OTHER_ROBOT: 'X'
    }
    
    obs = env.reset()
    
    def update(frame):
        ax.clear()
        
        # Action aléatoire pour la démo
        action = np.random.randint(0, env.action_space)
        obs, reward, done, info = env.step(action)
        
        if done:
            env.reset()
        
        # Créer l'image
        display = env.grid.copy()
        
        # Marquer les colis
        for pkg in env.packages:
            if not pkg.picked_up and not pkg.delivered:
                py, px = pkg.pickup_pos
                display[py, px] = CellType.PACKAGE
        
        # Marquer les autres robots
        for oy, ox in env.other_robots:
            display[oy, ox] = CellType.OTHER_ROBOT
        
        # Marquer notre robot
        ry, rx = env.robot.position
        display[ry, rx] = CellType.ROBOT
        
        # Afficher
        cmap = plt.cm.colors.ListedColormap([
            'white',      # FLOOR
            'black',      # WALL
            'gray',       # SHELF
            'lightgreen', # PICKUP
            'lightblue',  # DROPOFF
            'yellow',     # CHARGER
            'blue',       # ROBOT
            'orange',     # PACKAGE
            'red'         # OTHER_ROBOT
        ])
        
        ax.imshow(display, cmap=cmap, origin='upper')
        ax.set_title(f'🤖 Robot d\'Entrepôt - Step {env.steps}\n' +
                    f'Battery: {env.robot.battery:.0f}% | ' +
                    f'Carrying: {"📦" if env.robot.carrying else "No"} | ' +
                    f'Delivered: {env.robot.packages_delivered}/3',
                    fontsize=12)
        ax.axis('off')
        
        # Légende
        legend_text = 'Actions: ↑↓←→ Move | P Pickup | D Drop | ⚡ Charge | W Wait'
        ax.text(0.5, -0.05, legend_text, transform=ax.transAxes, 
               ha='center', fontsize=10)
    
    anim = FuncAnimation(fig, update, frames=100, interval=200, repeat=True)
    plt.tight_layout()
    plt.show()


def visualize_architecture():
    """
    🏗️ VISUALISATION 6: Architecture du DQN Hiérarchique
    Schéma de l'architecture à deux niveaux.
    """
    print("\n🏗️ Génération du schéma d'architecture...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Architecture DQN Hiérarchique', fontsize=16, fontweight='bold', pad=20)
    
    # Boîtes
    def draw_box(x, y, w, h, color, label, sublabel=''):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + 0.1, label, ha='center', va='center', 
               fontweight='bold', fontsize=11)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.3, sublabel, ha='center', va='center', 
                   fontsize=9, style='italic')
    
    # Environment
    draw_box(0.5, 4, 2.5, 2, '#e8f5e9', 'Environment', 'GridWorld/Warehouse')
    
    # State
    draw_box(4, 4.5, 1.5, 1, '#fff3e0', 'State', 's_t')
    
    # High-Level Policy
    draw_box(6.5, 7, 3, 2, '#e3f2fd', 'High-Level Policy', 'Q_high(s) → subgoal')
    
    # Subgoal
    draw_box(6.5, 4.5, 3, 1, '#fce4ec', 'Subgoal g', 'key/door/goal')
    
    # Low-Level Policy
    draw_box(6.5, 1.5, 3, 2, '#f3e5f5', 'Low-Level Policy', 'Q_low(s,g) → action')
    
    # Action
    draw_box(10.5, 4.5, 1.5, 1, '#fff8e1', 'Action', 'a_t')
    
    # Episodic Memory
    draw_box(11, 7, 2.5, 2, '#e8eaf6', 'Episodic\nMemory', '')
    
    # PER Buffer
    draw_box(11, 1.5, 2.5, 2, '#efebe9', 'PER\nBuffer', '')
    
    # Flèches
    arrow_props = dict(arrowstyle='->', color='black', lw=2)
    
    # Env -> State
    ax.annotate('', xy=(4, 5), xytext=(3, 5), arrowprops=arrow_props)
    
    # State -> High-Level
    ax.annotate('', xy=(6.5, 7.5), xytext=(5.5, 5.5), arrowprops=arrow_props)
    
    # High-Level -> Subgoal
    ax.annotate('', xy=(8, 5.5), xytext=(8, 7), arrowprops=arrow_props)
    
    # State -> Low-Level
    ax.annotate('', xy=(6.5, 2.5), xytext=(5.5, 4.5), arrowprops=arrow_props)
    
    # Subgoal -> Low-Level
    ax.annotate('', xy=(8, 3.5), xytext=(8, 4.5), arrowprops=arrow_props)
    
    # Low-Level -> Action
    ax.annotate('', xy=(10.5, 5), xytext=(9.5, 3), arrowprops=arrow_props)
    
    # Action -> Env
    ax.annotate('', xy=(1.75, 6), xytext=(11, 5.5), 
               arrowprops=dict(arrowstyle='->', color='green', lw=2, 
                              connectionstyle='arc3,rad=0.3'))
    
    # Memory connections
    ax.annotate('', xy=(11, 8), xytext=(9.5, 8), 
               arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
    ax.annotate('', xy=(11, 2.5), xytext=(9.5, 2.5),
               arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))
    
    # Annotations
    ax.text(7, 0.5, 'Le High-Level choisit les sous-objectifs (toutes les N steps)',
           ha='center', fontsize=10, style='italic', color='gray')
    ax.text(7, 9.5, 'Le Low-Level exécute les actions primitives (chaque step)',
           ha='center', fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig('results/plots/architecture_diagram.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Graphique sauvegardé: results/plots/architecture_diagram.png")


def visualize_heatmap():
    """
    🔥 VISUALISATION 7: Heatmap des Visites
    Montre les zones les plus visitées par l'agent.
    """
    print("\n🔥 Génération de la heatmap des visites...")
    
    env = GridWorld(size=10, max_steps=200)
    visit_counts = np.zeros((10, 10))
    
    # Simuler plusieurs épisodes
    for _ in range(100):
        obs = env.reset()
        done = False
        while not done:
            action = np.random.randint(0, env.action_space)
            obs, _, done, _ = env.step(action)
            y, x, _ = env.state
            visit_counts[y, x] += 1
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap
    ax = axes[0]
    im = ax.imshow(visit_counts, cmap='hot', interpolation='nearest')
    ax.set_title('Fréquence de Visite des Cases')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='Nombre de visites')
    
    # Grille avec annotations
    ax = axes[1]
    grid = env.grid.copy()
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'red', 'yellow', 'brown', 'green'])
    ax.imshow(grid, cmap=cmap)
    
    # Ajouter les compteurs
    for i in range(10):
        for j in range(10):
            if visit_counts[i, j] > 0:
                ax.text(j, i, f'{int(visit_counts[i, j])}', ha='center', va='center',
                       fontsize=8, color='blue' if grid[i, j] == 0 else 'white')
    
    ax.set_title('Grille avec Compteurs de Visite')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    plt.tight_layout()
    plt.savefig('results/plots/visit_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Graphique sauvegardé: results/plots/visit_heatmap.png")


def main():
    """Menu principal des visualisations."""
    print("=" * 60)
    print("       ProRL - Visualisations pour Présentation")
    print("=" * 60)
    
    # Créer le dossier plots si nécessaire
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--live':
            visualize_warehouse_live()
            return
        elif sys.argv[1] == '--all':
            pass
    
    print("\nQuelle visualisation voulez-vous générer?")
    print("  1. Courbes d'apprentissage comparatives")
    print("  2. Analyse théorique des synergies")
    print("  3. Résultats du robot d'entrepôt")
    print("  4. Trajectoire GridWorld")
    print("  5. Démonstration live du robot")
    print("  6. Schéma d'architecture")
    print("  7. Heatmap des visites")
    print("  8. TOUT générer")
    print("  0. Quitter")
    
    try:
        choice = input("\nVotre choix (1-8): ").strip()
    except:
        choice = '8'
    
    if choice == '1':
        plot_learning_curves()
    elif choice == '2':
        plot_theoretical_analysis()
    elif choice == '3':
        plot_warehouse_results()
    elif choice == '4':
        visualize_gridworld_trajectory()
    elif choice == '5':
        visualize_warehouse_live()
    elif choice == '6':
        visualize_architecture()
    elif choice == '7':
        visualize_heatmap()
    elif choice == '8':
        plot_learning_curves()
        plot_theoretical_analysis()
        plot_warehouse_results()
        visualize_gridworld_trajectory()
        visualize_architecture()
        visualize_heatmap()
        print("\n✅ Toutes les visualisations ont été générées!")
        print("📁 Fichiers dans: results/plots/")
    
    print("\n" + "=" * 60)
    print("Visualisations terminées!")


if __name__ == "__main__":
    main()
