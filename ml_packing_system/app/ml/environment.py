"""RL environment for tree packing optimization."""

import numpy as np
import torch
from typing import List, Tuple, Optional
from ..geometry import ChristmasTree, calculate_score, check_all_collisions
from ..state import PuzzleState


class PackingEnv:
    """Environment for RL-based packing optimization."""
    
    def __init__(self, puzzle_state: PuzzleState, max_steps: int = 100):
        """
        Initialize environment.
        
        Args:
            puzzle_state: Initial puzzle state
            max_steps: Maximum steps per episode
        """
        self.initial_state = puzzle_state.copy()
        self.current_state = puzzle_state.copy()
        self.max_steps = max_steps
        self.current_step = 0
        self.best_score = puzzle_state.score
        self.episode_reward = 0.0
        
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation
        """
        self.current_state = self.initial_state.copy()
        self.current_step = 0
        self.episode_reward = 0.0
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation.
        
        Returns:
            State vector: [tree_positions (x,y,deg), bounding_box_features]
        """
        trees = self.current_state.trees
        n = len(trees)
        
        # Flatten tree positions (normalized)
        positions = []
        for tree in trees:
            positions.extend([
                tree.x / 10.0,  # Normalize to ~[-10, 10]
                tree.y / 10.0,
                tree.deg / 360.0
            ])
        
        # Add global features
        side = self.current_state.side_length
        score = self.current_state.score
        
        global_features = [
            side / 10.0,
            score / 10.0,
            n / 200.0,
            self.current_step / self.max_steps
        ]
        
        # Pad positions to max size (200 trees * 3)
        max_position_dims = 200 * 3
        positions.extend([0.0] * (max_position_dims - len(positions)))
        
        obs = np.array(positions + global_features, dtype=np.float32)
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action array [dx, dy, ddeg] for each tree
        
        Returns:
            (observation, reward, done, info) tuple
        """
        self.current_step += 1
        
        # Apply action to trees
        action = action.reshape(-1, 3)
        prev_score = self.current_state.score
        
        for i, tree in enumerate(self.current_state.trees):
            if i < len(action):
                dx, dy, ddeg = action[i]
                tree.move(dx, dy, ddeg)
        
        # Check for collisions
        collisions = check_all_collisions(self.current_state.trees)
        has_collision = len(collisions) > 0
        
        # Update metrics
        self.current_state.update_metrics()
        new_score = self.current_state.score
        
        # Calculate reward
        reward = self._calculate_reward(prev_score, new_score, has_collision)
        self.episode_reward += reward
        
        # Check if done
        done = self.current_step >= self.max_steps or has_collision
        
        # Info dict
        info = {
            'score': new_score,
            'side_length': self.current_state.side_length,
            'collisions': len(collisions),
            'improvement': prev_score - new_score,
            'episode_reward': self.episode_reward
        }
        
        # Update best score
        if new_score < self.best_score and not has_collision:
            self.best_score = new_score
            info['new_best'] = True
        
        return self._get_observation(), reward, done, info
    
    def _calculate_reward(self, prev_score: float, new_score: float, has_collision: bool) -> float:
        """
        Calculate reward for the step.
        
        Args:
            prev_score: Previous score
            new_score: New score
            has_collision: Whether collision occurred
        
        Returns:
            Reward value
        """
        if has_collision:
            return -10.0  # Heavy penalty for collision
        
        # Reward for improvement (negative score change is good)
        improvement = prev_score - new_score
        
        if improvement > 0:
            # Scaled reward for improvement
            reward = improvement * 100.0
        else:
            # Small penalty for making things worse
            reward = improvement * 10.0
        
        # Bonus for being below best score
        if new_score < self.best_score:
            reward += 1.0
        
        return float(reward)
    
    def get_state(self) -> PuzzleState:
        """Get current puzzle state."""
        return self.current_state.copy()
    
    def set_state(self, state: PuzzleState):
        """Set current puzzle state."""
        self.current_state = state.copy()
        self.current_state.update_metrics()


def create_state_tensor(puzzle_state: PuzzleState, device: str = 'cpu') -> torch.Tensor:
    """
    Convert puzzle state to tensor for RL agent.
    
    Args:
        puzzle_state: Puzzle state to convert
        device: Device for tensor
    
    Returns:
        State tensor
    """
    env = PackingEnv(puzzle_state)
    obs = env._get_observation()
    return torch.from_numpy(obs).unsqueeze(0).to(device)
