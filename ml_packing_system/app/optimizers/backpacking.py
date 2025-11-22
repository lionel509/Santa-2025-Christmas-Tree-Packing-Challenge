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
from ..geometry import ChristmasTree, calculate_score


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
                adapted_trees = [t.copy() for t in best_puzzle.trees[:n]]
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
