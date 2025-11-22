"""Geometry module for tree packing calculations."""

from .tree import ChristmasTree
from .collision import check_collision, check_all_collisions, check_collision_optimized
from .bounds import calculate_bounding_square, calculate_score, get_square_bounds

__all__ = [
    'ChristmasTree',
    'check_collision',
    'check_all_collisions',
    'check_collision_optimized',
    'calculate_bounding_square',
    'calculate_score',
    'get_square_bounds'
]
