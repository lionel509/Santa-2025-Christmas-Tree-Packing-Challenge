"""Collision detection using Separating Axis Theorem (SAT)."""

import numpy as np
from typing import List
from .tree import ChristmasTree


def get_polygon_edges(polygon: np.ndarray) -> np.ndarray:
    """
    Get edges of a polygon.
    
    Args:
        polygon: Nx2 array of polygon vertices
    
    Returns:
        Nx2 array of edge vectors
    """
    return np.roll(polygon, -1, axis=0) - polygon


def get_perpendicular_axes(edges: np.ndarray) -> np.ndarray:
    """
    Get perpendicular axes (normals) for edges.
    
    Args:
        edges: Nx2 array of edge vectors
    
    Returns:
        Nx2 array of normalized perpendicular vectors
    """
    # Perpendicular: (x, y) -> (-y, x)
    perp = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
    # Normalize
    norms = np.linalg.norm(perp, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    return perp / norms


def project_polygon(polygon: np.ndarray, axis: np.ndarray) -> tuple:
    """
    Project polygon onto an axis.
    
    Args:
        polygon: Nx2 array of vertices
        axis: 2D normalized vector
    
    Returns:
        (min_proj, max_proj) tuple
    """
    projections = polygon @ axis
    return float(projections.min()), float(projections.max())


def check_collision_sat(poly1: np.ndarray, poly2: np.ndarray, tolerance: float = 1e-9) -> bool:
    """
    Check if two polygons overlap using Separating Axis Theorem.
    
    Args:
        poly1: First polygon vertices (Nx2)
        poly2: Second polygon vertices (Mx2)
        tolerance: Overlap tolerance (for touching detection)
    
    Returns:
        True if polygons overlap (not just touching)
    """
    # Get edges and perpendicular axes
    edges1 = get_polygon_edges(poly1)
    edges2 = get_polygon_edges(poly2)
    
    axes1 = get_perpendicular_axes(edges1)
    axes2 = get_perpendicular_axes(edges2)
    
    # Test all axes
    all_axes = np.vstack([axes1, axes2])
    
    for axis in all_axes:
        min1, max1 = project_polygon(poly1, axis)
        min2, max2 = project_polygon(poly2, axis)
        
        # Check for separation
        if max1 < min2 - tolerance or max2 < min1 - tolerance:
            return False  # Separating axis found, no collision
    
    return True  # No separating axis found, collision detected


def check_collision(tree1: ChristmasTree, tree2: ChristmasTree, tolerance: float = 1e-9) -> bool:
    """
    Check if two trees collide.
    
    Args:
        tree1: First tree
        tree2: Second tree
        tolerance: Overlap tolerance
    
    Returns:
        True if trees overlap
    """
    return check_collision_sat(tree1.polygon, tree2.polygon, tolerance)


def check_all_collisions(trees: List[ChristmasTree], tolerance: float = 1e-9) -> List[tuple]:
    """
    Check all pairwise collisions in a list of trees.
    
    Args:
        trees: List of trees
        tolerance: Overlap tolerance
    
    Returns:
        List of (i, j) tuples indicating colliding tree pairs
    """
    collisions = []
    n = len(trees)
    
    for i in range(n):
        for j in range(i + 1, n):
            if check_collision(trees[i], trees[j], tolerance):
                collisions.append((i, j))
    
    return collisions


def quick_bounds_overlap(tree1: ChristmasTree, tree2: ChristmasTree, margin: float = 0.0) -> bool:
    """
    Quick bounding box overlap check (faster pre-filter before SAT).
    
    Args:
        tree1: First tree
        tree2: Second tree
        margin: Extra margin for bounding box
    
    Returns:
        True if bounding boxes overlap
    """
    minx1, miny1, maxx1, maxy1 = tree1.get_bounds()
    minx2, miny2, maxx2, maxy2 = tree2.get_bounds()
    
    return not (maxx1 + margin < minx2 or 
                maxx2 + margin < minx1 or 
                maxy1 + margin < miny2 or 
                maxy2 + margin < miny1)


def check_collision_optimized(tree1: ChristmasTree, tree2: ChristmasTree, tolerance: float = 1e-9) -> bool:
    """
    Optimized collision check with bounding box pre-filter.
    
    Args:
        tree1: First tree
        tree2: Second tree
        tolerance: Overlap tolerance
    
    Returns:
        True if trees overlap
    """
    # Quick bounds check first
    if not quick_bounds_overlap(tree1, tree2):
        return False
    
    # Full SAT check
    return check_collision_sat(tree1.polygon, tree2.polygon, tolerance)
