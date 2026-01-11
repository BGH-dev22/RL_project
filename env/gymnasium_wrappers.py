"""
Gymnasium Environment Wrappers
==============================
Wrappers pour intégrer les environnements Atari et MuJoCo via Gymnasium.

Auteur: ProRL Project
Date: 2025
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np

try:
    import gymnasium as gym
    from gymnasium.wrappers import (
        FrameStackObservation,
        GrayscaleObservation,
        ResizeObservation,
        NormalizeObservation,
        NormalizeReward,
        ClipAction
    )
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    print("Warning: gymnasium not installed. Run: pip install gymnasium[atari,mujoco]")


# =============================================================================
# ATARI WRAPPER
# =============================================================================

class AtariWrapper:
    """
    Wrapper pour environnements Atari avec prétraitement standard:
    - Frame stacking (4 frames)
    - Grayscale conversion
    - Resize to 84x84
    - Frame skipping
    
    Environnements supportés: Breakout, Pong, SpaceInvaders, etc.
    """
    
    SUPPORTED_GAMES = [
        'ALE/Breakout-v5',
        'ALE/Pong-v5',
        'ALE/SpaceInvaders-v5',
        'ALE/Qbert-v5',
        'ALE/MsPacman-v5',
        'ALE/Asteroids-v5',
        'ALE/BeamRider-v5',
        'ALE/Enduro-v5',
        'ALE/Seaquest-v5'
    ]
    
    def __init__(
        self,
        game_name: str = 'ALE/Breakout-v5',
        frame_skip: int = 4,
        frame_stack: int = 4,
        image_size: int = 84,
        seed: Optional[int] = None
    ):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium is required. Run: pip install gymnasium[atari]")
        
        # Create base environment
        self.env = gym.make(
            game_name,
            frameskip=frame_skip,
            render_mode='rgb_array'
        )
        
        # Apply wrappers
        self.env = GrayscaleObservation(self.env)
        self.env = ResizeObservation(self.env, (image_size, image_size))
        self.env = FrameStackObservation(self.env, stack_size=frame_stack)
        
        if seed is not None:
            self.env.reset(seed=seed)
        
        self.action_dim = self.env.action_space.n
        self.obs_shape = self.env.observation_space.shape
        self.game_name = game_name
        
    @property
    def obs_dim(self) -> int:
        """Flattened observation dimension for MLP networks."""
        return int(np.prod(self.obs_shape))
    
    def reset(self) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset()
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
    
    def close(self):
        self.env.close()
    
    @classmethod
    def list_games(cls) -> list:
        return cls.SUPPORTED_GAMES


# =============================================================================
# MUJOCO WRAPPER
# =============================================================================

class MuJoCoWrapper:
    """
    Wrapper pour environnements MuJoCo avec normalisation:
    - Observation normalization
    - Action clipping
    - Reward scaling
    
    Environnements supportés: HalfCheetah, Ant, Walker2d, Hopper, etc.
    """
    
    SUPPORTED_ENVS = [
        'HalfCheetah-v5',
        'Ant-v5',
        'Walker2d-v5',
        'Hopper-v5',
        'Humanoid-v5',
        'Swimmer-v5',
        'InvertedPendulum-v5',
        'InvertedDoublePendulum-v5',
        'Reacher-v5',
        'Pusher-v5'
    ]
    
    def __init__(
        self,
        env_name: str = 'HalfCheetah-v5',
        normalize_obs: bool = True,
        normalize_reward: bool = True,
        seed: Optional[int] = None
    ):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium is required. Run: pip install gymnasium[mujoco]")
        
        # Create base environment
        self.env = gym.make(env_name, render_mode='rgb_array')
        
        # Apply wrappers
        self.env = ClipAction(self.env)
        
        if normalize_obs:
            self.env = NormalizeObservation(self.env)
        if normalize_reward:
            self.env = NormalizeReward(self.env)
        
        if seed is not None:
            self.env.reset(seed=seed)
        
        self.action_dim = self.env.action_space.shape[0]
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_high = self.env.action_space.high[0]
        self.action_low = self.env.action_space.low[0]
        self.env_name = env_name
        self.is_continuous = True
        
    def reset(self) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset()
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
    
    def close(self):
        self.env.close()
    
    @classmethod
    def list_envs(cls) -> list:
        return cls.SUPPORTED_ENVS


# =============================================================================
# UNIFIED ENVIRONMENT INTERFACE
# =============================================================================

class UnifiedEnv:
    """
    Interface unifiée pour tous les environnements (GridWorld, Atari, MuJoCo).
    Permet de switch facilement entre les environnements pour benchmarking.
    """
    
    def __init__(
        self,
        env_type: str,
        env_name: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        Args:
            env_type: 'gridworld', 'warehouse', 'atari', 'mujoco'
            env_name: Nom spécifique de l'environnement
            seed: Seed pour reproductibilité
        """
        self.env_type = env_type
        self.seed = seed
        
        if env_type == 'gridworld':
            from env.gridworld import GridWorld
            self.env = GridWorld(**kwargs)
            self.obs_dim = self.env.obs_dim
            self.action_dim = self.env.action_dim
            self.is_continuous = False
            
        elif env_type == 'warehouse':
            from env.warehouse_robot import WarehouseEnv
            self.env = WarehouseEnv(**kwargs)
            self.obs_dim = self.env.obs_dim
            self.action_dim = self.env.action_dim
            self.is_continuous = False
            
        elif env_type == 'atari':
            self.env = AtariWrapper(
                game_name=env_name or 'ALE/Breakout-v5',
                seed=seed,
                **kwargs
            )
            self.obs_dim = self.env.obs_dim
            self.action_dim = self.env.action_dim
            self.is_continuous = False
            self.obs_shape = self.env.obs_shape  # Pour CNN
            
        elif env_type == 'mujoco':
            self.env = MuJoCoWrapper(
                env_name=env_name or 'HalfCheetah-v5',
                seed=seed,
                **kwargs
            )
            self.obs_dim = self.env.obs_dim
            self.action_dim = self.env.action_dim
            self.is_continuous = True
            
        else:
            raise ValueError(f"Unknown env_type: {env_type}. "
                           f"Available: gridworld, warehouse, atari, mujoco")
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        if self.env_type in ['atari', 'mujoco']:
            obs, _ = self.env.reset()
            return obs
        else:
            return self.env.reset()
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take action and return (obs, reward, done, info)."""
        if self.env_type in ['atari', 'mujoco']:
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            return obs, reward, done, info
        else:
            return self.env.step(action)
    
    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()
    
    @staticmethod
    def available_environments() -> Dict[str, list]:
        """List all available environments."""
        return {
            'gridworld': ['default'],
            'warehouse': ['default'],
            'atari': AtariWrapper.SUPPORTED_GAMES if HAS_GYMNASIUM else [],
            'mujoco': MuJoCoWrapper.SUPPORTED_ENVS if HAS_GYMNASIUM else []
        }


# =============================================================================
# CNN NETWORK FOR ATARI
# =============================================================================

class AtariCNN:
    """
    CNN architecture pour Atari (Nature DQN style).
    
    Architecture:
    - Conv 32 filters, 8x8, stride 4
    - Conv 64 filters, 4x4, stride 2
    - Conv 64 filters, 3x3, stride 1
    - FC 512
    - Output: action_dim
    """
    
    def __init__(self, input_shape: Tuple[int, ...], action_dim: int):
        import torch.nn as nn
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate conv output size
        import torch
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out_size = self.features(dummy).shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        
    def forward(self, x):
        import torch
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.float() / 255.0  # Normalize pixel values
        features = self.features(x)
        return self.fc(features)


if __name__ == "__main__":
    # Test availability
    print("Available environments:")
    for env_type, envs in UnifiedEnv.available_environments().items():
        print(f"  {env_type}: {len(envs)} environments")
