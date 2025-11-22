"""Bounding box and scoring calculations."""

import numpy as np
from typing import List
from .tree import ChristmasTree


def calculate_bounding_square(trees: List[ChristmasTree]) -> float:
    """
    Calculate the side length of the smallest square bounding box.
    
    Args:
        trees: List of trees
    
    Returns:
        Side length of bounding square
    """
    if not trees:
        return 0.0
    
    # Get all polygon vertices
    all_vertices = np.vstack([tree.polygon for tree in trees])
    
    # Calculate bounds
    minx = all_vertices[:, 0].min()
    miny = all_vertices[:, 1].min()
    maxx = all_vertices[:, 0].max()
    maxy = all_vertices[:, 1].max()
    
    # Square side is the maximum of width and height
    width = maxx - minx
    height = maxy - miny
    side = max(width, height)
    
    return float(side)


def calculate_score(trees: List[ChristmasTree]) -> float:
    """
    Calculate normalized score: s² / n where s is bounding square side.
    
    Args:
        trees: List of trees
    
    Returns:
        Normalized score
    """
    if not trees:
        return 0.0
    
    side = calculate_bounding_square(trees)
    n = len(trees)
    score = (side ** 2) / n
    
    return float(score)


def get_bounding_box(trees: List[ChristmasTree]) -> tuple:
    """
    Get the actual bounding box coordinates.
    
    Args:
        trees: List of trees
    
    Returns:
        (minx, miny, maxx, maxy) tuple
    """
    if not trees:
        return (0.0, 0.0, 0.0, 0.0)
    
    all_vertices = np.vstack([tree.polygon for tree in trees])
    
    return (
        float(all_vertices[:, 0].min()),
        float(all_vertices[:, 1].min()),
        float(all_vertices[:, 0].max()),
        float(all_vertices[:, 1].max())
    )


def get_square_bounds(trees: List[ChristmasTree]) -> tuple:
    """
    Get the square bounding box coordinates (centered on actual bounds).
    
    Args:
        trees: List of trees
    
    Returns:
        (square_x, square_y, side) tuple representing bottom-left corner and side
    """
    if not trees:
        return (0.0, 0.0, 0.0)
    
    minx, miny, maxx, maxy = get_bounding_box(trees)
    width = maxx - minx
    height = maxy - miny
    side = max(width, height)
    
    # Center the square on the bounding box
    if width >= height:
        square_x = minx
        square_y = miny - (side - height) / 2
    else:
        square_x = minx - (side - width) / 2
        square_y = miny
    
    return (float(square_x), float(square_y), float(side))


def validate_positions(trees: List[ChristmasTree], max_coord: float = 100.0) -> bool:
    """
    Validate that all tree coordinates are within allowed bounds.
    
    Args:
        trees: List of trees
        max_coord: Maximum absolute coordinate value
    
    Returns:
        True if all positions are valid
    """
    for tree in trees:
        if abs(tree.x) > max_coord or abs(tree.y) > max_coord:
            return False
    return True


def calculate_compactness(trees: List[ChristmasTree]) -> float:
    """
    Calculate compactness metric: ratio of actual area to bounding square area.
    
    Args:
        trees: List of trees
    
    Returns:
        Compactness ratio (higher is better, max 1.0)
    """
    if not trees:
        return 0.0
    
    # Approximate tree area (using simple triangle approximation)
    tree_area = 0.35  # Approximate area of one tree
    total_tree_area = len(trees) * tree_area
    
    side = calculate_bounding_square(trees)
    bounding_area = side ** 2
    
    if bounding_area == 0:
        return 0.0
    
    return min(total_tree_area / bounding_area, 1.0)
