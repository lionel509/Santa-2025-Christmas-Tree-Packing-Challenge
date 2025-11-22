"""Heuristic optimization methods."""

import random
import math
import numpy as np
from typing import List, Optional
from ..geometry import ChristmasTree, check_all_collisions, calculate_score
from ..state import PuzzleState


def local_search(
    state: PuzzleState,
    iterations: int = 100,
    step_size: float = 0.05,
    rotation_step: float = 5.0
) -> PuzzleState:
    """
    Local search optimization with small random moves.
    
    Args:
        state: Initial state
        iterations: Number of iterations
        step_size: Position step size
        rotation_step: Rotation step size
    
    Returns:
        Optimized state
    """
    current = state.copy()
    best = current.copy()
    best_score = current.score
    
    for _ in range(iterations):
        # Select random tree
        tree_idx = random.randint(0, len(current.trees) - 1)
        tree = current.trees[tree_idx]
        
        # Save original position
        orig_x, orig_y, orig_deg = tree.x, tree.y, tree.deg
        
        # Random move
        dx = random.uniform(-step_size, step_size)
        dy = random.uniform(-step_size, step_size)
        ddeg = random.uniform(-rotation_step, rotation_step)
        
        tree.move(dx, dy, ddeg)
        
        # Check collisions
        collisions = check_all_collisions(current.trees)
        
        if not collisions:
            current.update_metrics()
            
            if current.score < best_score:
                best = current.copy()
                best_score = current.score
        else:
            # Revert move
            tree.update_position(orig_x, orig_y, orig_deg)
    
    return best


def simulated_annealing(
    state: PuzzleState,
    iterations: int = 1000,
    initial_temp: float = 1.0,
    cooling_rate: float = 0.995
) -> PuzzleState:
    """
    Simulated annealing optimization.
    
    Args:
        state: Initial state
        iterations: Number of iterations
        initial_temp: Initial temperature
        cooling_rate: Temperature decay rate
    
    Returns:
        Optimized state
    """
    current = state.copy()
    best = current.copy()
    best_score = current.score
    
    temp = initial_temp
    
    for i in range(iterations):
        # Select random tree
        tree_idx = random.randint(0, len(current.trees) - 1)
        tree = current.trees[tree_idx]
        
        # Save original
        orig_x, orig_y, orig_deg = tree.x, tree.y, tree.deg
        
        # Random move (larger at high temperature)
        scale = temp * 0.5
        dx = random.gauss(0, scale)
        dy = random.gauss(0, scale)
        ddeg = random.gauss(0, 30 * temp)
        
        tree.move(dx, dy, ddeg)
        
        # Check collisions
        collisions = check_all_collisions(current.trees)
        
        if collisions:
            # Reject collision
            tree.update_position(orig_x, orig_y, orig_deg)
        else:
            # Calculate new score
            old_score = current.score
            current.update_metrics()
            new_score = current.score
            
            # Accept or reject based on Metropolis criterion
            delta = new_score - old_score
            
            if delta < 0 or random.random() < math.exp(-delta / (temp + 1e-10)):
                # Accept move
                if new_score < best_score:
                    best = current.copy()
                    best_score = new_score
            else:
                # Reject move
                tree.update_position(orig_x, orig_y, orig_deg)
                current.score = old_score
        
        # Cool down
        temp *= cooling_rate
    
    return best


def squeeze_bounds(state: PuzzleState, factor: float = 0.98) -> PuzzleState:
    """
    Squeeze trees towards center to reduce bounding box.
    
    Args:
        state: Initial state
        factor: Squeeze factor (< 1.0)
    
    Returns:
        Squeezed state (or original if collision)
    """
    squeezed = state.copy()
    
    # Calculate centroid
    cx = sum(tree.x for tree in squeezed.trees) / len(squeezed.trees)
    cy = sum(tree.y for tree in squeezed.trees) / len(squeezed.trees)
    
    # Move trees towards centroid
    for tree in squeezed.trees:
        dx = tree.x - cx
        dy = tree.y - cy
        new_x = cx + dx * factor
        new_y = cy + dy * factor
        tree.update_position(new_x, new_y, tree.deg)
    
    # Check collisions
    collisions = check_all_collisions(squeezed.trees)
    
    if collisions:
        return state  # Return original if collision
    
    squeezed.update_metrics()
    
    if squeezed.score < state.score:
        return squeezed
    else:
        return state


def rotation_sweep(state: PuzzleState, tree_idx: int, steps: int = 36) -> PuzzleState:
    """
    Try different rotations for a specific tree.
    
    Args:
        state: Current state
        tree_idx: Index of tree to rotate
        steps: Number of rotation steps to try
    
    Returns:
        Best state found
    """
    best = state.copy()
    best_score = state.score
    
    tree = state.trees[tree_idx]
    orig_deg = tree.deg
    
    for i in range(steps):
        test_state = state.copy()
        test_tree = test_state.trees[tree_idx]
        new_deg = (360.0 / steps) * i
        test_tree.update_position(test_tree.x, test_tree.y, new_deg)
        
        collisions = check_all_collisions(test_state.trees)
        
        if not collisions:
            test_state.update_metrics()
            if test_state.score < best_score:
                best = test_state
                best_score = test_state.score
    
    return best


def jitter_positions(
    state: PuzzleState,
    magnitude: float = 0.02,
    num_trees: Optional[int] = None
) -> PuzzleState:
    """
    Add small random jitter to tree positions.
    
    Args:
        state: Current state
        magnitude: Jitter magnitude
        num_trees: Number of trees to jitter (None = all)
    
    Returns:
        Jittered state (or original if collision)
    """
    jittered = state.copy()
    
    if num_trees is None:
        trees_to_jitter = list(range(len(jittered.trees)))
    else:
        trees_to_jitter = random.sample(range(len(jittered.trees)), min(num_trees, len(jittered.trees)))
    
    for idx in trees_to_jitter:
        tree = jittered.trees[idx]
        dx = random.uniform(-magnitude, magnitude)
        dy = random.uniform(-magnitude, magnitude)
        ddeg = random.uniform(-2, 2)
        tree.move(dx, dy, ddeg)
    
    collisions = check_all_collisions(jittered.trees)
    
    if collisions:
        return state
    
    jittered.update_metrics()
    
    if jittered.score < state.score:
        return jittered
    else:
        return state
