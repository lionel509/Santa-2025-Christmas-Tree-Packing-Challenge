"""Heuristic optimization methods."""

import random
import math
import numpy as np
from typing import List, Optional
from ..geometry import ChristmasTree, check_all_collisions, calculate_score, check_collision_optimized
from ..geometry.bounds import get_bounding_box
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


def iterative_squeeze(state: PuzzleState, step_size: float = 0.05, iterations: int = 1) -> PuzzleState:
    """
    Iteratively move individual trees towards the centroid.
    
    Args:
        state: Current state
        step_size: Movement step size
        iterations: Number of passes over all trees
    
    Returns:
        Squeezed state
    """
    current = state.copy()
    
    # Calculate centroid
    cx = sum(tree.x for tree in current.trees) / len(current.trees)
    cy = sum(tree.y for tree in current.trees) / len(current.trees)
    
    for _ in range(iterations):
        # Randomize order to avoid bias
        indices = list(range(len(current.trees)))
        random.shuffle(indices)
        
        for idx in indices:
            tree = current.trees[idx]
            orig_x, orig_y = tree.x, tree.y
            
            # Vector to centroid
            dx = cx - tree.x
            dy = cy - tree.y
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < 1e-6:
                continue
                
            # Normalize and scale
            move_x = (dx / dist) * step_size
            move_y = (dy / dist) * step_size
            
            # Move tree
            tree.move(move_x, move_y, 0)
            
            # Check collisions for this tree only
            collision = False
            for other_idx, other_tree in enumerate(current.trees):
                if idx != other_idx:
                    if check_collision_optimized(tree, other_tree):
                        collision = True
                        break
            
            if collision:
                # Revert
                tree.update_position(orig_x, orig_y, tree.deg)
    
    current.update_metrics()
    return current


def boundary_compression(state: PuzzleState, step_size: float = 0.05) -> PuzzleState:
    """
    Try to move boundary trees inwards.
    
    Args:
        state: Current state
        step_size: Movement step size
    
    Returns:
        Compressed state
    """
    current = state.copy()
    
    # Get bounds
    minx, miny, maxx, maxy = get_bounding_box(current.trees)
    width = maxx - minx
    height = maxy - miny
    
    # Determine which dimension to compress (the larger one determines the square side)
    # Actually, we should try to compress all sides, but prioritize the ones defining the max dimension
    
    # Identify boundary trees (within a small margin of the bounds)
    margin = 0.1
    
    # Directions to move: (dx, dy)
    moves = []
    
    # If width > height, compressing height doesn't help the score (side length) immediately, 
    # but might help later. However, score is max(width, height)^2.
    # So we should prioritize the larger dimension.
    
    if width >= height:
        # Compress X
        moves.append(('min_x', step_size, 0))
        moves.append(('max_x', -step_size, 0))
    else:
        # Compress Y
        moves.append(('min_y', 0, step_size))
        moves.append(('max_y', 0, -step_size))
        
    # Also try the other dimension with lower probability or just do it?
    # Let's just try all 4 sides, but maybe larger steps for the critical dimension?
    # For now, let's just try to push everything in from the edges.
    
    # Define boundary groups
    groups = {
        'min_x': [], 'max_x': [], 'min_y': [], 'max_y': []
    }
    
    for idx, tree in enumerate(current.trees):
        t_minx, t_miny, t_maxx, t_maxy = tree.get_bounds()
        if abs(t_minx - minx) < margin: groups['min_x'].append(idx)
        if abs(t_maxx - maxx) < margin: groups['max_x'].append(idx)
        if abs(t_miny - miny) < margin: groups['min_y'].append(idx)
        if abs(t_maxy - maxy) < margin: groups['max_y'].append(idx)
    
    # Try to move trees in groups
    for group_name, indices in groups.items():
        dx, dy = 0.0, 0.0
        if group_name == 'min_x': dx = step_size
        elif group_name == 'max_x': dx = -step_size
        elif group_name == 'min_y': dy = step_size
        elif group_name == 'max_y': dy = -step_size
        
        for idx in indices:
            tree = current.trees[idx]
            orig_x, orig_y = tree.x, tree.y
            
            tree.move(dx, dy, 0)
            
            # Check collisions
            collision = False
            for other_idx, other_tree in enumerate(current.trees):
                if idx != other_idx:
                    if check_collision_optimized(tree, other_tree):
                        collision = True
                        break
            
            if collision:
                tree.update_position(orig_x, orig_y, tree.deg)
                
    current.update_metrics()
    return current


def scramble_positions(state: PuzzleState, magnitude: float = 0.5) -> PuzzleState:
    """
    Aggressively scramble tree positions to escape local optima.
    
    Args:
        state: Current state
        magnitude: Scramble magnitude
    
    Returns:
        Scrambled state (validity not guaranteed, collision resolution needed after)
    """
    scrambled = state.copy()
    
    # Better approach: Randomize positions completely but keep them somewhat compact
    n = len(scrambled.trees)
    side = math.sqrt(n) * 1.5 # Rough estimate
    
    for tree in scrambled.trees:
        tree.update_position(
            random.uniform(0, side),
            random.uniform(0, side),
            random.uniform(0, 360)
        )
        
    # Simple collision resolution
    # This is a naive "drop" approach
    resolved_trees = []
    for tree in scrambled.trees:
        # Try to place tree
        placed = False
        for _ in range(10): # Try 10 random spots
            if not any(check_collision_optimized(tree, other) for other in resolved_trees):
                resolved_trees.append(tree)
                placed = True
                break
            # Move slightly
            tree.move(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-10, 10))
        
        if not placed:
            # If can't place, just append (will have collisions)
            # But we need valid state.
            # Let's just place it far away
            tree.update_position(side * 2, len(resolved_trees) * 0.5, 0)
            resolved_trees.append(tree)
            
    scrambled.trees = resolved_trees
    scrambled.update_metrics()
    return scrambled

