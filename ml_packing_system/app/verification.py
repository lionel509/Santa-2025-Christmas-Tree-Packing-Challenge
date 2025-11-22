"""Verification module for validating puzzle solutions."""

import numpy as np
from typing import List, Dict, Any
from .geometry import (
    ChristmasTree,
    check_all_collisions,
    calculate_minimum_gap,
    get_all_gaps,
    calculate_score
)
from .state import PuzzleState


def verify_puzzle(puzzle: PuzzleState, tolerance: float = 0.0) -> Dict[str, Any]:
    """
    Comprehensive verification of a puzzle solution.
    
    Args:
        puzzle: Puzzle state to verify
        tolerance: Collision tolerance (0.0 for strict zero-gap detection)
    
    Returns:
        Dictionary with verification results
    """
    trees = puzzle.trees
    n_trees = len(trees)
    
    # Check for collisions
    collisions = check_all_collisions(trees, tolerance=tolerance)
    has_collisions = len(collisions) > 0
    
    # Calculate all gaps
    gaps_data = get_all_gaps(trees)
    gaps = [g['gap'] for g in gaps_data]
    
    # Gap statistics
    if gaps:
        min_gap = float(np.min(gaps))
        max_gap = float(np.max(gaps))
        avg_gap = float(np.mean(gaps))
        median_gap = float(np.median(gaps))
        
        # Count gaps in ranges
        zero_gaps = sum(1 for g in gaps if g < 1e-6)
        tiny_gaps = sum(1 for g in gaps if 1e-6 <= g < 1e-3)
        small_gaps = sum(1 for g in gaps if 1e-3 <= g < 0.01)
        medium_gaps = sum(1 for g in gaps if 0.01 <= g < 0.1)
        large_gaps = sum(1 for g in gaps if g >= 0.1)
    else:
        min_gap = max_gap = avg_gap = median_gap = 0.0
        zero_gaps = tiny_gaps = small_gaps = medium_gaps = large_gaps = 0
    
    # Verify tree count matches puzzle number
    correct_tree_count = (n_trees == puzzle.n)
    
    # Verify all trees have valid positions
    valid_positions = all(
        not (np.isnan(tree.x) or np.isnan(tree.y) or np.isnan(tree.deg))
        for tree in trees
    )
    
    # Verify all trees have valid rotations (0-360)
    valid_rotations = all(0 <= tree.deg < 360 for tree in trees)
    
    # Calculate score
    score = calculate_score(trees)
    score_matches = abs(score - puzzle.score) < 1e-6
    
    # Overall validity
    is_valid = (
        not has_collisions and
        correct_tree_count and
        valid_positions and
        valid_rotations and
        score_matches
    )
    
    return {
        'puzzle_id': puzzle.n,
        'is_valid': is_valid,
        'has_collisions': has_collisions,
        'collision_count': len(collisions),
        'collisions': [{'tree_i': i, 'tree_j': j} for i, j in collisions[:10]],  # First 10
        'tree_count': n_trees,
        'expected_tree_count': puzzle.n,
        'correct_tree_count': correct_tree_count,
        'valid_positions': valid_positions,
        'valid_rotations': valid_rotations,
        'score': float(score),
        'puzzle_score': float(puzzle.score),
        'score_matches': score_matches,
        'gap_statistics': {
            'min': min_gap,
            'max': max_gap,
            'avg': avg_gap,
            'median': median_gap,
            'total_pairs': len(gaps),
            'zero_gaps': zero_gaps,
            'tiny_gaps': tiny_gaps,
            'small_gaps': small_gaps,
            'medium_gaps': medium_gaps,
            'large_gaps': large_gaps
        },
        'tolerance': tolerance
    }


def verify_all_puzzles(puzzles: List[PuzzleState], tolerance: float = 0.0) -> Dict[str, Any]:
    """
    Verify all puzzles and provide summary statistics.
    
    Args:
        puzzles: List of puzzle states
        tolerance: Collision tolerance (0.0 for strict zero-gap detection)
    
    Returns:
        Dictionary with overall verification results
    """
    results = []
    
    for puzzle in puzzles:
        result = verify_puzzle(puzzle, tolerance)
        results.append(result)
    
    # Summary statistics
    total_puzzles = len(results)
    valid_puzzles = sum(1 for r in results if r['is_valid'])
    puzzles_with_collisions = sum(1 for r in results if r['has_collisions'])
    total_collisions = sum(r['collision_count'] for r in results)
    
    # Gap statistics across all puzzles
    all_min_gaps = [r['gap_statistics']['min'] for r in results if r['gap_statistics']['total_pairs'] > 0]
    all_avg_gaps = [r['gap_statistics']['avg'] for r in results if r['gap_statistics']['total_pairs'] > 0]
    
    if all_min_gaps:
        global_min_gap = float(np.min(all_min_gaps))
        global_max_gap = float(np.max([r['gap_statistics']['max'] for r in results]))
        global_avg_gap = float(np.mean(all_avg_gaps))
    else:
        global_min_gap = global_max_gap = global_avg_gap = 0.0
    
    # Total zero/tiny gaps
    total_zero_gaps = sum(r['gap_statistics']['zero_gaps'] for r in results)
    total_tiny_gaps = sum(r['gap_statistics']['tiny_gaps'] for r in results)
    
    return {
        'summary': {
            'total_puzzles': total_puzzles,
            'valid_puzzles': valid_puzzles,
            'invalid_puzzles': total_puzzles - valid_puzzles,
            'puzzles_with_collisions': puzzles_with_collisions,
            'total_collisions': total_collisions,
            'validation_rate': float(valid_puzzles / total_puzzles) if total_puzzles > 0 else 0.0
        },
        'gap_summary': {
            'global_min_gap': global_min_gap,
            'global_max_gap': global_max_gap,
            'global_avg_gap': global_avg_gap,
            'total_zero_gaps': total_zero_gaps,
            'total_tiny_gaps': total_tiny_gaps
        },
        'puzzle_results': results,
        'tolerance': tolerance
    }


def get_puzzle_verification_status(puzzle: PuzzleState) -> str:
    """
    Get a simple status string for a puzzle.
    
    Args:
        puzzle: Puzzle state
    
    Returns:
        Status string: "VALID", "COLLISION", "INVALID"
    """
    result = verify_puzzle(puzzle, tolerance=0.0)
    
    if result['is_valid']:
        return "VALID"
    elif result['has_collisions']:
        return "COLLISION"
    else:
        return "INVALID"
