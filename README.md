# ProRL: DQN Hiérarchique avec Mémoire Épisodique et Explicabilité

Ce dépôt fournit un framework complet pour étudier une architecture DQN hiérarchique dotée d'une mémoire épisodique priorisée, d'un mécanisme "return-then-explore" et d'un module d'explicabilité par trajectoires.

## 🚀 Contributions Originales

### 1. Application Réelle: Robot d'Entrepôt 🤖
**Fichier:** `env/warehouse_robot.py`

Environnement réaliste inspiré d'Amazon Robotics:
- Robot mobile naviguant dans un entrepôt 20x15
- Ramassage et livraison de colis (multi-objectifs)
- Gestion de batterie (recharge nécessaire)
- Obstacles dynamiques (autres robots)
- 8 actions: UP, DOWN, LEFT, RIGHT, PICKUP, DROP, CHARGE, WAIT

**Résultats après 1000 épisodes:**
- Taux de mission complète: **18%**
- Colis livrés: **1.6/3** en moyenne
- Mort batterie: réduit de 100% → **12%**

### 2. Mémoire Épisodique Adaptative (AEM-CS)
**Fichier:** `agents/adaptive_episodic_memory.py`
- Similarité contextuelle (pas seulement spatiale)
- Clustering adaptatif des épisodes par pattern de succès
- Reconstruction de trajectoires optimales
- Meta-learning pour ajuster les paramètres de priorité

### 3. Analyse Théorique des Synergies
**Fichier:** `analysis/theoretical_analysis.py`
- Formalisation des synergies entre composants DQN
- Estimation des bornes de convergence
- Analyse de complexité d'échantillonnage
- Insights: Hiérarchique + Mémoire = synergie optimale

### 4. Transfer Learning
**Fichier:** `experiments/transfer_learning.py`
- Zero-shot transfer vers nouveaux environnements
- Few-shot fine-tuning avec features gelés
- Démonstration de généralisation des skills

## 📊 Résultats (GridWorld)

| Variante | Goals (%) | Convergence Relative |
|----------|-----------|---------------------|
| vanilla  | 66.6%     | 1.00x               |
| per      | 58.4%     | 1.25x               |
| memory   | 67.9%     | 1.33x               |
| hier     | 71.3%     | 1.43x               |
| **full** | **72.8%** | **4.55x**           |

## Structure
- `env/gridworld.py` — Environnement GridWorld (clé-porte-goal)
- `env/warehouse_robot.py` — **[NOUVEAU]** Environnement robotique réaliste
- `agents/dqn_base.py` — DQN standard avec rejouage priorisé
- `agents/hierarchical_dqn.py` — Architecture hiérarchique (high/low level)
- `agents/episodic_memory.py` — Mémoire épisodique standard
- `agents/adaptive_episodic_memory.py` — **[NOUVEAU]** Mémoire épisodique améliorée
- `experiments/train_warehouse.py` — **[NOUVEAU]** Entraînement robot d'entrepôt
- `experiments/compare_variants.py` — Comparaison des 6 variantes
- `experiments/transfer_learning.py` — **[NOUVEAU]** Expériences de transfert
- `analysis/theoretical_analysis.py` — **[NOUVEAU]** Analyse théorique

## Démarrage rapide
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install torch numpy matplotlib

# Comparer toutes les variantes (GridWorld)
python experiments/compare_variants.py

# Entraîner le robot d'entrepôt
python experiments/train_warehouse.py --episodes 1000

# Lancer les expériences d'innovation
python experiments/run_innovations.py
```

## Références Théoriques

- **DQN**: Mnih et al., 2015
- **PER**: Schaul et al., 2015  
- **Hierarchical RL**: Kulkarni et al., 2016
- **Episodic Memory**: Blundell et al., 2016
- **Warehouse Robotics**: Inspired by Amazon Robotics / Kiva Systems
