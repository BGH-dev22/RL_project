"""
Warehouse Robot Environment
===========================
Environnement réaliste de robotique mobile pour entrepôt.

Inspiré par les systèmes Amazon Robotics / Kiva Systems:
- Robot mobile navigue dans un entrepôt
- Ramasse des colis à différentes stations
- Livre à des zones de dépôt
- Évite les obstacles (statiques et dynamiques)
- Gestion de la batterie
- Multi-objectifs simultanés

Auteur: ProRL Project
Date: 2025
"""

import random
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from enum import IntEnum
from dataclasses import dataclass


class CellType(IntEnum):
    FLOOR = 0
    WALL = 1
    SHELF = 2        # Étagère (obstacle, source de colis)
    PICKUP = 3       # Zone de ramassage
    DROPOFF = 4      # Zone de dépôt
    CHARGER = 5      # Station de recharge
    ROBOT = 6        # Position du robot
    PACKAGE = 7      # Colis au sol
    OTHER_ROBOT = 8  # Autre robot (obstacle dynamique)


@dataclass
class Package:
    """Représente un colis à livrer."""
    id: int
    pickup_pos: Tuple[int, int]
    dropoff_pos: Tuple[int, int]
    priority: int  # 1-3, 3 = urgent
    picked_up: bool = False
    delivered: bool = False


@dataclass
class RobotState:
    """État complet du robot."""
    position: Tuple[int, int]
    battery: float  # 0-100
    carrying: Optional[Package] = None
    packages_delivered: int = 0
    total_distance: int = 0


class WarehouseEnv:
    """
    Environnement d'entrepôt robotisé.
    
    Actions:
    0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
    4: PICKUP (ramasser colis)
    5: DROP (déposer colis)
    6: CHARGE (se recharger)
    7: WAIT (attendre - pour éviter collisions)
    """
    
    ACTIONS = ["up", "down", "left", "right", "pickup", "drop", "charge", "wait"]
    
    def __init__(
        self,
        width: int = 20,
        height: int = 15,
        num_shelves: int = 30,
        num_packages: int = 5,
        num_other_robots: int = 2,
        max_steps: int = 500,
        battery_drain: float = 0.5,
        seed: Optional[int] = None
    ):
        self.width = width
        self.height = height
        self.num_shelves = num_shelves
        self.num_packages = num_packages
        self.num_other_robots = num_other_robots
        self.max_steps = max_steps
        self.battery_drain = battery_drain
        
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        
        self.grid: np.ndarray = None
        self.robot: RobotState = None
        self.packages: List[Package] = []
        self.other_robots: List[Tuple[int, int]] = []
        self.pickup_zones: List[Tuple[int, int]] = []
        self.dropoff_zones: List[Tuple[int, int]] = []
        self.charger_pos: Tuple[int, int] = None
        
        self.steps = 0
        self.total_reward = 0
        
        self._build_warehouse()
    
    def _build_warehouse(self) -> None:
        """Construire le layout de l'entrepôt."""
        self.grid = np.zeros((self.height, self.width), dtype=int)
        
        # Murs extérieurs
        self.grid[0, :] = CellType.WALL
        self.grid[-1, :] = CellType.WALL
        self.grid[:, 0] = CellType.WALL
        self.grid[:, -1] = CellType.WALL
        
        # Étagères en rangées (pattern réaliste d'entrepôt)
        shelf_rows = [3, 4, 7, 8, 11, 12]
        for row in shelf_rows:
            if row < self.height - 1:
                for col in range(3, self.width - 3, 3):
                    if col < self.width - 1:
                        self.grid[row, col] = CellType.SHELF
                        self.grid[row, col + 1] = CellType.SHELF
        
        # Zones de ramassage (côté gauche)
        self.pickup_zones = [(2, 2), (5, 2), (9, 2)]
        for y, x in self.pickup_zones:
            if 0 < y < self.height - 1 and 0 < x < self.width - 1:
                self.grid[y, x] = CellType.PICKUP
        
        # Zones de dépôt (côté droit)
        self.dropoff_zones = [(2, self.width - 3), (6, self.width - 3), (10, self.width - 3)]
        for y, x in self.dropoff_zones:
            if 0 < y < self.height - 1 and 0 < x < self.width - 1:
                self.grid[y, x] = CellType.DROPOFF
        
        # Station de recharge (coin)
        self.charger_pos = (self.height - 2, self.width - 2)
        self.grid[self.charger_pos] = CellType.CHARGER
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Réinitialiser l'environnement."""
        self._build_warehouse()
        self.steps = 0
        self.total_reward = 0
        
        # Position initiale du robot (coin supérieur gauche)
        start_pos = (1, 1)
        self.robot = RobotState(
            position=start_pos,
            battery=100.0,
            carrying=None,
            packages_delivered=0,
            total_distance=0
        )
        
        # Générer des colis
        self.packages = []
        for i in range(self.num_packages):
            pickup = self.rng.choice(self.pickup_zones)
            dropoff = self.rng.choice(self.dropoff_zones)
            priority = self.rng.randint(1, 3)
            self.packages.append(Package(
                id=i,
                pickup_pos=pickup,
                dropoff_pos=dropoff,
                priority=priority
            ))
        
        # Positionner les autres robots (obstacles dynamiques)
        self.other_robots = []
        for _ in range(self.num_other_robots):
            while True:
                pos = (self.rng.randint(1, self.height - 2),
                      self.rng.randint(1, self.width - 2))
                if self.grid[pos] == CellType.FLOOR and pos != start_pos:
                    self.other_robots.append(pos)
                    break
        
        return self._get_observation()
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Obtenir l'observation actuelle."""
        y, x = self.robot.position
        
        # État du robot (vecteur compact)
        robot_state = np.array([
            y / self.height,  # Position normalisée
            x / self.width,
            self.robot.battery / 100.0,
            1.0 if self.robot.carrying else 0.0,
            self.robot.packages_delivered / max(1, self.num_packages),
        ], dtype=np.float32)
        
        # Distances aux objectifs importants
        distances = self._compute_distances()
        
        # Vision locale (5x5 autour du robot)
        local_view = self._get_local_view(radius=2)
        
        # Informations sur les colis
        package_info = self._get_package_info()
        
        # Observation combinée
        full_obs = np.concatenate([
            robot_state,
            distances,
            package_info
        ])
        
        return {
            "state": full_obs,
            "local_view": local_view,
            "grid": self.grid.copy()
        }
    
    def _compute_distances(self) -> np.ndarray:
        """Calculer les distances aux objectifs importants."""
        y, x = self.robot.position
        
        # Distance au colis le plus proche non ramassé
        dist_to_pickup = 1.0
        for pkg in self.packages:
            if not pkg.picked_up and not pkg.delivered:
                py, px = pkg.pickup_pos
                d = (abs(y - py) + abs(x - px)) / (self.height + self.width)
                dist_to_pickup = min(dist_to_pickup, d)
        
        # Distance à la zone de dépôt (si on porte un colis)
        dist_to_dropoff = 1.0
        if self.robot.carrying:
            dy, dx = self.robot.carrying.dropoff_pos
            dist_to_dropoff = (abs(y - dy) + abs(x - dx)) / (self.height + self.width)
        
        # Distance au chargeur
        cy, cx = self.charger_pos
        dist_to_charger = (abs(y - cy) + abs(x - cx)) / (self.height + self.width)
        
        # Distance au robot le plus proche
        dist_to_other = 1.0
        for oy, ox in self.other_robots:
            d = (abs(y - oy) + abs(x - ox)) / (self.height + self.width)
            dist_to_other = min(dist_to_other, d)
        
        return np.array([
            dist_to_pickup,
            dist_to_dropoff,
            dist_to_charger,
            dist_to_other
        ], dtype=np.float32)
    
    def _get_local_view(self, radius: int = 2) -> np.ndarray:
        """Obtenir une vue locale autour du robot."""
        y, x = self.robot.position
        size = 2 * radius + 1
        local = np.full((size, size), CellType.WALL, dtype=int)
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    local[dy + radius, dx + radius] = self.grid[ny, nx]
        
        return local.flatten().astype(np.float32) / 8.0  # Normalisé
    
    def _get_package_info(self) -> np.ndarray:
        """Informations sur les colis."""
        # Nombre de colis par état
        pending = sum(1 for p in self.packages if not p.picked_up and not p.delivered)
        in_transit = sum(1 for p in self.packages if p.picked_up and not p.delivered)
        delivered = sum(1 for p in self.packages if p.delivered)
        
        # Priorité moyenne des colis restants
        remaining = [p for p in self.packages if not p.delivered]
        avg_priority = np.mean([p.priority for p in remaining]) if remaining else 0
        
        return np.array([
            pending / max(1, self.num_packages),
            in_transit / max(1, self.num_packages),
            delivered / max(1, self.num_packages),
            avg_priority / 3.0
        ], dtype=np.float32)
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """Exécuter une action."""
        self.steps += 1
        old_pos = self.robot.position
        reward = -0.1  # Coût temporel
        info = {}
        
        # Drain de batterie
        self.robot.battery -= self.battery_drain
        if self.robot.battery <= 0:
            self.robot.battery = 0
            reward -= 50  # Pénalité sévère: robot bloqué
            info["battery_dead"] = True
            return self._get_observation(), reward, True, info
        
        # Mouvement
        y, x = self.robot.position
        new_y, new_x = y, x
        
        if action == 0:  # UP
            new_y = y - 1
        elif action == 1:  # DOWN
            new_y = y + 1
        elif action == 2:  # LEFT
            new_x = x - 1
        elif action == 3:  # RIGHT
            new_x = x + 1
        elif action == 4:  # PICKUP
            reward += self._try_pickup()
        elif action == 5:  # DROP
            reward += self._try_drop()
            info["drop_attempt"] = True
        elif action == 6:  # CHARGE
            reward += self._try_charge()
        elif action == 7:  # WAIT
            reward -= 0.05  # Léger coût d'attente
        
        # Vérifier si le mouvement est valide
        if action < 4:
            if self._is_valid_move(new_y, new_x):
                self.robot.position = (new_y, new_x)
                self.robot.total_distance += 1
                
                # Bonus pour se rapprocher de l'objectif
                reward += self._compute_progress_reward(old_pos, (new_y, new_x))
            else:
                reward -= 1  # Pénalité pour collision
                info["collision"] = True
        
        # Déplacer les autres robots (obstacles dynamiques)
        self._move_other_robots()
        
        # Vérifier si tous les colis sont livrés
        all_delivered = all(p.delivered for p in self.packages)
        if all_delivered:
            reward += 100  # Gros bonus
            info["mission_complete"] = True
            return self._get_observation(), reward, True, info
        
        # Timeout
        done = self.steps >= self.max_steps
        if done:
            info["timeout"] = True
        
        self.total_reward += reward
        return self._get_observation(), reward, done, info
    
    def _is_valid_move(self, y: int, x: int) -> bool:
        """Vérifier si un mouvement est valide."""
        if not (0 < y < self.height - 1 and 0 < x < self.width - 1):
            return False
        if self.grid[y, x] in [CellType.WALL, CellType.SHELF]:
            return False
        if (y, x) in self.other_robots:
            return False
        return True
    
    def _try_pickup(self) -> float:
        """Tenter de ramasser un colis."""
        if self.robot.carrying:
            return -1  # Déjà un colis
        
        y, x = self.robot.position
        for pkg in self.packages:
            if not pkg.picked_up and not pkg.delivered and pkg.pickup_pos == (y, x):
                pkg.picked_up = True
                self.robot.carrying = pkg
                return 10 + pkg.priority * 2  # Bonus selon priorité
        
        return -0.5  # Pas de colis ici
    
    def _try_drop(self) -> float:
        """Tenter de déposer un colis."""
        if not self.robot.carrying:
            return -1  # Pas de colis
        
        y, x = self.robot.position
        pkg = self.robot.carrying
        
        if pkg.dropoff_pos == (y, x):
            pkg.delivered = True
            self.robot.carrying = None
            self.robot.packages_delivered += 1
            return 30 + pkg.priority * 5  # Gros bonus
        
        return -2  # Mauvais endroit
    
    def _try_charge(self) -> float:
        """Tenter de recharger."""
        if self.robot.position != self.charger_pos:
            return -1  # Pas au chargeur
        
        old_battery = self.robot.battery
        self.robot.battery = min(100, self.robot.battery + 20)
        charged = self.robot.battery - old_battery
        
        return charged * 0.1  # Petit bonus pour recharge
    
    def _compute_progress_reward(self, old_pos: Tuple[int, int], 
                                  new_pos: Tuple[int, int]) -> float:
        """Reward shaping basé sur le progrès."""
        oy, ox = old_pos
        ny, nx = new_pos
        reward = 0.0
        
        if self.robot.carrying:
            # Se rapprocher du dropoff
            dy, dx = self.robot.carrying.dropoff_pos
            old_dist = abs(oy - dy) + abs(ox - dx)
            new_dist = abs(ny - dy) + abs(nx - dx)
            reward += (old_dist - new_dist) * 0.5
        else:
            # Se rapprocher d'un colis
            for pkg in self.packages:
                if not pkg.picked_up and not pkg.delivered:
                    py, px = pkg.pickup_pos
                    old_dist = abs(oy - py) + abs(ox - px)
                    new_dist = abs(ny - py) + abs(nx - px)
                    reward += (old_dist - new_dist) * 0.3
                    break  # Un seul colis à la fois
        
        # Pénalité si batterie faible et loin du chargeur
        if self.robot.battery < 30:
            cy, cx = self.charger_pos
            old_dist = abs(oy - cy) + abs(ox - cx)
            new_dist = abs(ny - cy) + abs(nx - cx)
            reward += (old_dist - new_dist) * 0.3
        
        return reward
    
    def _move_other_robots(self) -> None:
        """Déplacer les autres robots aléatoirement."""
        new_positions = []
        for oy, ox in self.other_robots:
            if self.rng.random() < 0.3:  # 30% chance de bouger
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                self.rng.shuffle(directions)
                moved = False
                for dy, dx in directions:
                    ny, nx = oy + dy, ox + dx
                    if (self._is_valid_move(ny, nx) and 
                        (ny, nx) != self.robot.position and
                        (ny, nx) not in new_positions):
                        new_positions.append((ny, nx))
                        moved = True
                        break
                if not moved:
                    new_positions.append((oy, ox))
            else:
                new_positions.append((oy, ox))
        self.other_robots = new_positions
    
    def render_text(self) -> str:
        """Rendu texte de l'environnement."""
        display = self.grid.copy()
        
        # Marquer les colis
        for pkg in self.packages:
            if not pkg.picked_up and not pkg.delivered:
                py, px = pkg.pickup_pos
                display[py, px] = CellType.PACKAGE
        
        # Marquer les autres robots
        for oy, ox in self.other_robots:
            display[oy, ox] = CellType.OTHER_ROBOT
        
        # Marquer notre robot
        ry, rx = self.robot.position
        display[ry, rx] = CellType.ROBOT
        
        symbols = {
            CellType.FLOOR: '.',
            CellType.WALL: '#',
            CellType.SHELF: '▓',
            CellType.PICKUP: 'P',
            CellType.DROPOFF: 'D',
            CellType.CHARGER: '⚡',
            CellType.ROBOT: '🤖' if not self.robot.carrying else '📦',
            CellType.PACKAGE: '□',
            CellType.OTHER_ROBOT: 'X'
        }
        
        lines = []
        for row in display:
            lines.append(' '.join(symbols.get(c, '?') for c in row))
        
        # Info status
        lines.append(f"\nBattery: {self.robot.battery:.0f}% | " +
                    f"Carrying: {'Yes' if self.robot.carrying else 'No'} | " +
                    f"Delivered: {self.robot.packages_delivered}/{self.num_packages} | " +
                    f"Steps: {self.steps}")
        
        return '\n'.join(lines)
    
    @property
    def action_space(self) -> int:
        return len(self.ACTIONS)
    
    @property
    def observation_dim(self) -> int:
        """Dimension de l'observation vectorielle."""
        return 5 + 4 + 4 + 25  # robot_state + distances + package_info + local_view
    
    @property
    def obs_dim(self) -> int:
        """Alias for observation_dim for compatibility."""
        return self.observation_dim
    
    @property
    def action_dim(self) -> int:
        """Alias for action_space for compatibility."""
        return self.action_space
    
    def get_state_vector(self, obs: Dict) -> np.ndarray:
        """Convertir observation en vecteur pour le réseau."""
        return np.concatenate([obs["state"], obs["local_view"]])


class WarehouseSubgoals:
    """Sous-objectifs pour l'architecture hiérarchique."""
    
    SUBGOALS = [
        "go_to_package",    # 0: Aller chercher un colis
        "pickup_package",   # 1: Ramasser le colis
        "go_to_dropoff",    # 2: Aller à la zone de dépôt
        "drop_package",     # 3: Déposer le colis
        "go_to_charger",    # 4: Aller au chargeur
        "charge",           # 5: Se recharger
        "avoid_collision"   # 6: Éviter collision
    ]
    
    @staticmethod
    def get_current_subgoal(robot: RobotState, packages: List[Package]) -> int:
        """Déterminer le sous-objectif actuel basé sur l'état."""
        # Priorité 1: Batterie critique
        if robot.battery < 20:
            if robot.position == (14, 18):  # Position charger (approximative)
                return 5  # charge
            return 4  # go_to_charger
        
        # Priorité 2: Livrer si on porte un colis
        if robot.carrying:
            if robot.position == robot.carrying.dropoff_pos:
                return 3  # drop_package
            return 2  # go_to_dropoff
        
        # Priorité 3: Aller chercher un colis
        pending = [p for p in packages if not p.picked_up and not p.delivered]
        if pending:
            # Vérifier si on est sur une zone de pickup
            for pkg in pending:
                if robot.position == pkg.pickup_pos:
                    return 1  # pickup_package
            return 0  # go_to_package
        
        return 0  # Par défaut
    
    @staticmethod
    def get_subgoal_reward(subgoal: int, info: Dict, robot: RobotState) -> float:
        """Calculer le reward intrinsèque pour un sous-objectif."""
        if subgoal == 0 and info.get("got_closer_to_package"):
            return 1.0
        if subgoal == 1 and info.get("pickup_success"):
            return 5.0
        if subgoal == 2 and info.get("got_closer_to_dropoff"):
            return 1.0
        if subgoal == 3 and robot.packages_delivered > 0:
            return 10.0
        if subgoal == 4 and info.get("got_closer_to_charger"):
            return 1.0
        if subgoal == 5 and robot.battery > 80:
            return 3.0
        return 0.0
