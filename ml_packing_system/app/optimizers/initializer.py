"""Initial layout generation."""

import random
import math
import numpy as np
from typing import List, Optional
from ..geometry import ChristmasTree, check_collision_optimized, calculate_score
from ..state import PuzzleState


def random_initialization(n: int) -> PuzzleState:
    """
    Create initial random layout for n trees.
    
    Args:
        n: Number of trees
    
    Returns:
        Initial puzzle state
    """
    trees = []
    
    for i in range(n):
        # Random position and rotation
        angle = random.uniform(0, 360)
        radius = random.uniform(0, 3.0 * math.sqrt(n))
        theta = random.uniform(0, 2 * math.pi)
        
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        
        tree = ChristmasTree(x, y, angle)
        trees.append(tree)
    
    state = PuzzleState(
        n=n,
        trees=trees,
        score=0.0,
        side_length=0.0
    )
    state.update_metrics()
    
    return state


def greedy_placement(n: int, base_trees: Optional[List[ChristmasTree]] = None) -> PuzzleState:
    """
    Greedy placement algorithm (similar to competition starter code).
    
    Args:
        n: Number of trees
        base_trees: Optional base trees to build upon
    
    Returns:
        Puzzle state with greedily placed trees
    """
    if base_trees is None:
        placed = []
    else:
        placed = [tree.copy() for tree in base_trees[:n]]
    
    num_to_add = n - len(placed)
    
    # Place first tree at origin if starting from scratch
    if not placed and num_to_add > 0:
        placed.append(ChristmasTree(0, 0, random.uniform(0, 360)))
        num_to_add -= 1
    
    # Place remaining trees
    for _ in range(num_to_add):
        best_tree = None
        best_radius = float('inf')
        
        # Try multiple random placements
        for attempt in range(10):
            # Random angle and rotation
            angle = random.uniform(0, 2 * math.pi)
            rotation = random.uniform(0, 360)
            
            # Start far from center
            radius = 20.0
            step_in = 0.5
            
            # Move towards center until collision
            collision_found = False
            while radius >= 0:
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                candidate = ChristmasTree(x, y, rotation)
                
                # Check collisions with placed trees
                has_collision = any(
                    check_collision_optimized(candidate, tree)
                    for tree in placed
                )
                
                if has_collision:
                    collision_found = True
                    break
                
                radius -= step_in
            
            # Back up if collision found
            if collision_found:
                step_out = 0.05
                while True:
                    radius += step_out
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    candidate = ChristmasTree(x, y, rotation)
                    
                    has_collision = any(
                        check_collision_optimized(candidate, tree)
                        for tree in placed
                    )
                    
                    if not has_collision:
                        break
            else:
                radius = 0
            
            # Keep best placement
            if radius < best_radius:
                best_radius = radius
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                best_tree = ChristmasTree(x, y, rotation)
        
        if best_tree is not None:
            placed.append(best_tree)
    
    state = PuzzleState(
        n=n,
        trees=placed,
        score=0.0,
        side_length=0.0
    )
    state.update_metrics()
    
    return state


def initialize_puzzle(n: int, method: str = 'greedy', base_trees: Optional[List[ChristmasTree]] = None) -> PuzzleState:
    """
    Initialize puzzle with n trees using specified method.
    
    Args:
        n: Number of trees
        method: Initialization method ('random', 'greedy')
        base_trees: Optional base trees for incremental building
    
    Returns:
        Initial puzzle state
    """
    if method == 'random':
        return random_initialization(n)
    elif method == 'greedy':
        return greedy_placement(n, base_trees)
    else:
        raise ValueError(f"Unknown initialization method: {method}")


def initialize_all_puzzles(method: str = 'greedy') -> dict:
    """
    Initialize all puzzles (1-200 trees) incrementally.
    
    Args:
        method: Initialization method
    
    Returns:
        Dictionary mapping n -> PuzzleState
    """
    puzzles = {}
    base_trees = None
    
    for n in range(1, 201):
        if method == 'greedy' and base_trees is not None:
            # Build incrementally
            puzzle = greedy_placement(n, base_trees)
        else:
            puzzle = initialize_puzzle(n, method)
        
        puzzles[n] = puzzle
        base_trees = puzzle.trees
        
        if n % 20 == 0:
            print(f"Initialized puzzle {n}/200 (score: {puzzle.score:.4f})")
    
    return puzzles
