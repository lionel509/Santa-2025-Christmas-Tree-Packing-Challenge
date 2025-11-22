"""Backward Iteration (BackPacking) heuristic optimizer.

This module implements the backward iteration approach described in the
notebook: starting from the largest configuration (200 trees), tracking
the best bounding square side length, and propagating that layout
backwards by truncating trees when a smaller-n configuration has a worse
side length. This yields consistent, compact layouts across all n.

Integration notes:
- Uses existing `ChristmasTree` geometry class and scoring utilities.
- Accepts a dictionary mapping n -> PuzzleState (e.g. from PuzzleManager).
- Produces a new dictionary of optimized PuzzleState objects plus an
  improvements list for analysis.

The algorithm expects each PuzzleState to have correctly updated
`side_length` and `score` fields (manager/state should handle this).
"""

from typing import Dict, List, Tuple
from ..state import PuzzleState
from ..geometry import ChristmasTree, calculate_score, calculate_bounding_square
from ..geometry.bounds import get_bounding_box


class BackPackingResult:
    """Container for BackPacking optimization outputs."""

    def __init__(self, puzzles: Dict[int, PuzzleState], improvements: List[dict], total_score: float):
        self.puzzles = puzzles              # Optimized puzzles mapping n -> PuzzleState
        self.improvements = improvements    # List of improvement dicts
        self.total_score = total_score      # Sum of scores across all n

    def summary(self) -> dict:
        return {
            "total_score": self.total_score,
            "improvements_count": len(self.improvements),
            "avg_improvement_pct": (
                sum(d["improvement_pct"] for d in self.improvements) / len(self.improvements)
                if self.improvements else 0.0
            ),
        }


def _compute_side_length(puzzle: PuzzleState) -> float:
    """Return current puzzle side length (already stored)."""
    return float(puzzle.side_length)


def smart_truncate(trees: List[ChristmasTree], target_n: int) -> List[ChristmasTree]:
    """
    Truncate tree list to target_n by iteratively removing boundary trees
    that minimize the bounding box.
    """
    current_trees = [t.copy() for t in trees]
    
    while len(current_trees) > target_n:
        # Get current bounds
        minx, miny, maxx, maxy = get_bounding_box(current_trees)
        
        # Find trees on the boundary
        candidates = []
        margin = 1e-4
        for i, t in enumerate(current_trees):
            t_minx, t_miny, t_maxx, t_maxy = t.get_bounds()
            if (abs(t_minx - minx) < margin or abs(t_maxx - maxx) < margin or
                abs(t_miny - miny) < margin or abs(t_maxy - maxy) < margin):
                candidates.append(i)
        
        if not candidates:
            # Fallback: remove last
            current_trees.pop()
            continue
            
        # Try removing each candidate
        best_idx = -1
        best_new_side = float('inf')
        
        for idx in candidates:
            # Create temp list without this tree
            # Optimization: don't copy full list, just pass iterator to calc bounds?
            # calculate_bounding_square takes list.
            temp_trees = current_trees[:idx] + current_trees[idx+1:]
            new_side = calculate_bounding_square(temp_trees)
            
            if new_side < best_new_side:
                best_new_side = new_side
                best_idx = idx
        
        # Remove the best candidate
        if best_idx != -1:
            current_trees.pop(best_idx)
        else:
            current_trees.pop()
            
    return current_trees


def run_backpacking(puzzles: Dict[int, PuzzleState]) -> BackPackingResult:
    """Run backward iteration on provided puzzle states.

    Args:
        puzzles: Mapping of n (1..200) to PuzzleState (may be partial; missing n skipped)

    Returns:
        BackPackingResult with optimized puzzles and improvement stats.
    """
    if not puzzles:
        return BackPackingResult({}, [], 0.0)

    optimized: Dict[int, PuzzleState] = {}
    improvements: List[dict] = []

    best_side = float("inf")
    best_puzzle: PuzzleState | None = None

    # Iterate from largest available n downwards
    for n in sorted(puzzles.keys(), reverse=True):
        puzzle = puzzles[n].copy()
        current_side = _compute_side_length(puzzle)

        if current_side < best_side:
            # New best layout encountered
            best_side = current_side
            best_puzzle = puzzle.copy()
            optimized[n] = puzzle
        else:
            # Adapt from best by truncating trees if possible
            if best_puzzle is not None and len(best_puzzle.trees) >= n:
                # Use smart truncation instead of simple slicing
                adapted_trees = smart_truncate(best_puzzle.trees, n)
                
                adapted_state = PuzzleState(n=n, trees=adapted_trees, score=0.0, side_length=0.0)
                adapted_state.update_metrics()

                adapted_side = _compute_side_length(adapted_state)
                if adapted_side < current_side:
                    improvement_pct = ((current_side - adapted_side) / current_side) * 100.0 if current_side > 0 else 0.0
                    improvements.append({
                        "n": n,
                        "original_side": current_side,
                        "optimized_side": adapted_side,
                        "improvement_pct": improvement_pct,
                    })
                    optimized[n] = adapted_state
                else:
                    optimized[n] = puzzle
            else:
                optimized[n] = puzzle

    # Now ensure forward fill (1..max) ordering for missing n (unlikely if full set provided)
    for n in range(1, 201):
        if n not in optimized and n in puzzles:
            optimized[n] = puzzles[n].copy()

    # Compute total score sum across all optimized puzzles
    total_score = 0.0
    for n, puzzle in optimized.items():
        # Recalculate score to be safe (already updated_metrics above, but ensures consistency)
        puzzle.update_metrics()
        total_score += puzzle.score

    return BackPackingResult(optimized, improvements, total_score)


def apply_backpacking(manager) -> BackPackingResult:  # type: ignore
    """Helper to run BackPacking directly from a PuzzleManager instance.

    Args:
        manager: PuzzleManager with existing puzzles

    Returns:
        BackPackingResult
    """
    puzzles = {n: manager.get_puzzle(n) for n in range(1, 201) if manager.get_puzzle(n) is not None}
    return run_backpacking(puzzles)


__all__ = [
    "BackPackingResult",
    "run_backpacking",
    "apply_backpacking",
]
