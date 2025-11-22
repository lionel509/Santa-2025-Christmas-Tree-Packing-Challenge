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


def check_collision_sat(poly1: np.ndarray, poly2: np.ndarray, tolerance: float = 0.0) -> bool:
    """
    Check if two polygons overlap using Separating Axis Theorem.
    
    Args:
        poly1: First polygon vertices (Nx2)
        poly2: Second polygon vertices (Mx2)
        tolerance: Overlap tolerance - set to 0.0 for ZERO gap (strict collision detection)
    
    Returns:
        True if polygons overlap (including touching)
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
        
        # Check for separation - NO TOLERANCE (strict)
        # If max1 <= min2 or max2 <= min1, they are separated or just touching
        if max1 <= min2 - tolerance or max2 <= min1 - tolerance:
            return False  # Separating axis found, no collision
    
    return True  # No separating axis found, collision detected


def check_collision(tree1: ChristmasTree, tree2: ChristmasTree, tolerance: float = 0.0) -> bool:
    """
    Check if two trees collide.
    
    Args:
        tree1: First tree
        tree2: Second tree
        tolerance: Overlap tolerance (0.0 for strict zero-gap collision detection)
    
    Returns:
        True if trees overlap
    """
    return check_collision_sat(tree1.polygon, tree2.polygon, tolerance)


def check_all_collisions(trees: List[ChristmasTree], tolerance: float = 0.0) -> List[tuple]:
    """
    Check all pairwise collisions in a list of trees.
    
    Args:
        trees: List of trees
        tolerance: Overlap tolerance (0.0 for strict zero-gap collision detection)
    
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


def check_collision_optimized(tree1: ChristmasTree, tree2: ChristmasTree, tolerance: float = 0.0) -> bool:
    """
    Optimized collision check with bounding box pre-filter.
    
    Args:
        tree1: First tree
        tree2: Second tree
        tolerance: Overlap tolerance (0.0 for strict zero-gap collision detection)
    
    Returns:
        True if trees overlap
    """
    # Quick bounds check first
    if not quick_bounds_overlap(tree1, tree2):
        return False
    
    # Full SAT check - STRICT (no tolerance)
    return check_collision_sat(tree1.polygon, tree2.polygon, tolerance)


def calculate_minimum_gap(tree1: ChristmasTree, tree2: ChristmasTree) -> float:
    """
    Calculate the minimum gap/distance between two trees.
    Uses point-to-polygon distance calculation.
    
    Args:
        tree1: First tree
        tree2: Second tree
    
    Returns:
        Minimum gap distance (0 if touching/overlapping, positive if separated)
    """
    poly1 = tree1.polygon
    poly2 = tree2.polygon
    
    # Calculate minimum distance between all vertex pairs
    min_dist = float('inf')
    
    # Distance from poly1 vertices to poly2 edges
    for vertex in poly1:
        for i in range(len(poly2)):
            edge_start = poly2[i]
            edge_end = poly2[(i + 1) % len(poly2)]
            
            # Distance from point to line segment
            dist = point_to_segment_distance(vertex, edge_start, edge_end)
            min_dist = min(min_dist, dist)
    
    # Distance from poly2 vertices to poly1 edges
    for vertex in poly2:
        for i in range(len(poly1)):
            edge_start = poly1[i]
            edge_end = poly1[(i + 1) % len(poly1)]
            
            dist = point_to_segment_distance(vertex, edge_start, edge_end)
            min_dist = min(min_dist, dist)
    
    # If overlapping, distance is 0
    if check_collision_sat(poly1, poly2, 0.0):
        return 0.0
    
    return float(min_dist)


def point_to_segment_distance(point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
    """
    Calculate distance from a point to a line segment.
    
    Args:
        point: Point coordinates
        seg_start: Segment start point
        seg_end: Segment end point
    
    Returns:
        Minimum distance
    """
    # Vector from segment start to end
    seg_vec = seg_end - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)
    
    if seg_len_sq < 1e-12:
        # Degenerate segment (point)
        return float(np.linalg.norm(point - seg_start))
    
    # Project point onto segment line
    t = max(0, min(1, np.dot(point - seg_start, seg_vec) / seg_len_sq))
    
    # Closest point on segment
    closest = seg_start + t * seg_vec
    
    return float(np.linalg.norm(point - closest))


def get_all_gaps(trees: List[ChristmasTree]) -> List[dict]:
    """
    Get all pairwise gaps between trees.
    
    Args:
        trees: List of trees
    
    Returns:
        List of dicts with gap information: {'i': int, 'j': int, 'gap': float}
    """
    gaps = []
    n = len(trees)
    
    for i in range(n):
        for j in range(i + 1, n):
            gap = calculate_minimum_gap(trees[i], trees[j])
            gaps.append({'i': i, 'j': j, 'gap': gap})
    
    return gaps

