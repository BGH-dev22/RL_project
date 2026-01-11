import random
from typing import Dict, List, Optional, Tuple
import numpy as np

# Simple 2D GridWorld with keys, doors, traps, and sparse rewards.
class GridWorld:
    ACTIONS = ["up", "down", "left", "right", "use_key"]

    def __init__(self, size: int = 10, max_steps: int = 500, partial_observability: bool = False, seed: Optional[int] = None):
        self.size = size
        self.max_steps = max_steps
        self.partial_observability = partial_observability
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.state: Tuple[int, int, bool] = (0, 0, False)
        self.steps = 0
        self.grid = np.zeros((size, size), dtype=int)
        self.goal: Tuple[int, int] = (size - 2, size - 2)
        self.traps: List[Tuple[int, int]] = []
        self.walls: List[Tuple[int, int]] = []
        self.keys: List[Tuple[int, int]] = []
        self.door: Tuple[int, int] = (size // 2, size // 2)
        self._build_layout()

    def _build_layout(self) -> None:
        # Create a wall barrier with a door in the middle - goal is behind the barrier
        mid = self.size // 2
        # Horizontal wall barrier across the grid
        self.walls = [(mid, j) for j in range(self.size) if j != mid]
        # Door is in the middle of the barrier
        self.door = (mid, mid)
        # Goal is behind the barrier
        self.goal = (self.size - 2, self.size - 2)
        # Key is on the starting side
        self.keys = [(1, self.size - 2)]
        # Traps scattered on both sides
        self.traps = [(2, 2), (mid - 1, 1), (mid + 2, mid + 2)]
        
        for y, x in self.walls:
            self.grid[y, x] = 1
        for y, x in self.traps:
            if 0 <= y < self.size and 0 <= x < self.size:
                self.grid[y, x] = 2
        for y, x in self.keys:
            self.grid[y, x] = 3
        dy, dx = self.door
        self.grid[dy, dx] = 4
        gy, gx = self.goal
        self.grid[gy, gx] = 5

    def reset(self) -> Dict[str, np.ndarray]:
        self.steps = 0
        self.state = (0, 0, False)  # (y, x, has_key)
        self.door_opened = False
        self._rebuild_grid()
        return self._obs()

    def _rebuild_grid(self) -> None:
        """Rebuild grid for fresh episode."""
        self.grid = np.zeros((self.size, self.size), dtype=int)
        for y, x in self.walls:
            self.grid[y, x] = 1
        for y, x in self.traps:
            self.grid[y, x] = 2
        for y, x in self.keys:
            self.grid[y, x] = 3
        dy, dx = self.door
        self.grid[dy, dx] = 4
        gy, gx = self.goal
        self.grid[gy, gx] = 5

    def get_current_subgoal(self) -> Tuple[int, int]:
        """Return current sub-objective based on state."""
        y, x, has_key = self.state
        if not has_key:
            # Go to nearest key
            return self.keys[0] if self.keys else self.door
        elif not self.door_opened:
            return self.door
        else:
            return self.goal

    def shaped_reward(self, old_state: Tuple[int, int, bool], new_state: Tuple[int, int, bool]) -> float:
        """Compute reward shaping based on progress toward sub-goal."""
        subgoal = self.get_current_subgoal()
        old_dist = abs(old_state[0] - subgoal[0]) + abs(old_state[1] - subgoal[1])
        new_dist = abs(new_state[0] - subgoal[0]) + abs(new_state[1] - subgoal[1])
        # Reward for getting closer, penalty for moving away
        shaping = (old_dist - new_dist) * 2.0
        # Bonus hint: if at door with key, encourage staying to use_key
        if new_state[2] and (new_state[0], new_state[1]) == self.door and not self.door_opened:
            shaping += 5.0  # hint bonus for being at door with key
        return shaping

    def _obs(self) -> Dict[str, np.ndarray]:
        y, x, has_key = self.state
        if self.partial_observability:
            radius = 2
            y0, y1 = max(0, y - radius), min(self.size, y + radius + 1)
            x0, x1 = max(0, x - radius), min(self.size, x + radius + 1)
            local = self.grid[y0:y1, x0:x1].copy()
            padding = ((radius - (y - y0), radius - (y1 - 1 - y)), (radius - (x - x0), radius - (x1 - 1 - x)))
            if padding[0][0] > 0 or padding[0][1] > 0 or padding[1][0] > 0 or padding[1][1] > 0:
                local = np.pad(local, padding, constant_values=-1)
            return {"agent": np.array([y, x, int(has_key)], dtype=np.int32), "local": local}
        return {"agent": np.array([y, x, int(has_key)], dtype=np.int32), "grid": self.grid.copy()}

    def _is_wall(self, y: int, x: int) -> bool:
        return (y, x) in self.walls

    def _in_bounds(self, y: int, x: int) -> bool:
        return 0 <= y < self.size and 0 <= x < self.size

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        self.steps += 1
        old_state = self.state
        y, x, has_key = self.state
        reward = -0.5  # reduced time penalty
        info: Dict[str, object] = {}

        if action == 0:  # up
            new_y, new_x = y - 1, x
        elif action == 1:  # down
            new_y, new_x = y + 1, x
        elif action == 2:  # left
            new_y, new_x = y, x - 1
        elif action == 3:  # right
            new_y, new_x = y, x + 1
        else:  # use_key
            new_y, new_x = y, x
            if (y, x) == self.door and has_key and not self.door_opened:
                self.door_opened = True
                self.grid[y, x] = 0
                info["door_opened"] = True
                reward += 20  # bonus for opening door
        if action != 4:
            if self._in_bounds(new_y, new_x) and not self._is_wall(new_y, new_x):
                # Allow stepping onto door cell (to use key there), but can't pass through to other side
                if (new_y, new_x) == self.door:
                    # Can always step onto door cell
                    y, x = new_y, new_x
                else:
                    # Check if trying to pass through closed door (coming from door to other side)
                    y, x = new_y, new_x
        cell = self.grid[y, x]
        if cell == 2:
            reward -= 30  # reduced trap penalty
        if cell == 3:
            has_key = True
            self.grid[y, x] = 0
            reward += 15  # bonus for getting key
            info["got_key"] = True
        if cell == 5:
            reward += 100
            done = True
            self.state = (y, x, has_key)
            info["goal_reached"] = True
            return self._obs(), reward, done, info
        
        self.state = (y, x, has_key)
        # Add shaped reward for progress toward sub-goal
        reward += self.shaped_reward(old_state, self.state)
        
        done = self.steps >= self.max_steps
        return self._obs(), reward, done, info

    def render_text(self) -> str:
        y, x, has_key = self.state
        grid = self.grid.copy()
        grid[y, x] = 9
        return "\n".join(" ".join(f"{cell:2d}" for cell in row) for row in grid)

    @property
    def action_space(self) -> int:
        return len(self.ACTIONS)

    @property
    def observation_space(self) -> int:
        return 3  # y, x, has_key encoded; grid/local is separate
    
    @property
    def obs_dim(self) -> int:
        """Alias for observation_space for compatibility."""
        return self.observation_space
    
    @property
    def action_dim(self) -> int:
        """Alias for action_space for compatibility."""
        return self.action_space
