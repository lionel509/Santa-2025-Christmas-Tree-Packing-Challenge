"""Geometry module for tree packing calculations."""

from .tree import ChristmasTree
from .collision import (
    check_collision, 
    check_all_collisions, 
    check_collision_optimized,
    calculate_minimum_gap,
    get_all_gaps
)
from .bounds import calculate_bounding_square, calculate_score, get_square_bounds

__all__ = [
    'ChristmasTree',
    'check_collision',
    'check_all_collisions',
    'check_collision_optimized',
    'calculate_minimum_gap',
    'get_all_gaps',
    'calculate_bounding_square',
    'calculate_score',
    'get_square_bounds'
]
