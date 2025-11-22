"""Puzzle state management."""

import json
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from ..geometry import ChristmasTree, calculate_bounding_square, calculate_score


@dataclass
class PuzzleState:
    """State for a single puzzle (n trees)."""
    n: int  # Number of trees
    trees: List[ChristmasTree]
    score: float
    side_length: float
    iterations: int = 0
    last_improvement: float = 0.0
    collisions: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'n': self.n,
            'trees': [tree.to_dict() for tree in self.trees],
            'score': self.score,
            'side_length': self.side_length,
            'iterations': self.iterations,
            'last_improvement': self.last_improvement,
            'collisions': self.collisions
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PuzzleState':
        """Create from dictionary."""
        trees = [ChristmasTree.from_dict(t) for t in data['trees']]
        return cls(
            n=data['n'],
            trees=trees,
            score=data['score'],
            side_length=data['side_length'],
            iterations=data.get('iterations', 0),
            last_improvement=data.get('last_improvement', 0.0),
            collisions=data.get('collisions', 0)
        )
    
    def update_metrics(self):
        """Recalculate score and side length."""
        self.side_length = calculate_bounding_square(self.trees)
        self.score = calculate_score(self.trees)
    
    def copy(self) -> 'PuzzleState':
        """Create a deep copy of this state."""
        return PuzzleState(
            n=self.n,
            trees=[tree.copy() for tree in self.trees],
            score=self.score,
            side_length=self.side_length,
            iterations=self.iterations,
            last_improvement=self.last_improvement,
            collisions=self.collisions
        )


class PuzzleManager:
    """Manages all puzzle states (1-200 trees)."""
    
    def __init__(self):
        self.puzzles: Dict[int, PuzzleState] = {}
        self.total_score: float = 0.0
        self.creation_time: float = time.time()
        self.last_save_time: float = time.time()
        
    def add_puzzle(self, puzzle: PuzzleState):
        """Add or update a puzzle state."""
        self.puzzles[puzzle.n] = puzzle
        self._update_total_score()
    
    def get_puzzle(self, n: int) -> Optional[PuzzleState]:
        """Get puzzle state for n trees."""
        return self.puzzles.get(n)
    
    def _update_total_score(self):
        """Recalculate total score across all puzzles."""
        self.total_score = sum(p.score for p in self.puzzles.values())
    
    def get_all_puzzles(self) -> List[PuzzleState]:
        """Get all puzzle states sorted by n."""
        return [self.puzzles[n] for n in sorted(self.puzzles.keys())]
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        if not self.puzzles:
            return {
                'total_puzzles': 0,
                'total_score': 0.0,
                'avg_score': 0.0,
                'total_iterations': 0,
                'uptime_seconds': time.time() - self.creation_time
            }
        
        total_iterations = sum(p.iterations for p in self.puzzles.values())
        avg_score = self.total_score / len(self.puzzles)
        
        return {
            'total_puzzles': len(self.puzzles),
            'total_score': self.total_score,
            'avg_score': avg_score,
            'total_iterations': total_iterations,
            'uptime_seconds': time.time() - self.creation_time,
            'last_save_seconds_ago': time.time() - self.last_save_time
        }
    
    def to_dict(self) -> dict:
        """Serialize all puzzles."""
        return {
            'puzzles': {n: p.to_dict() for n, p in self.puzzles.items()},
            'total_score': self.total_score,
            'creation_time': self.creation_time,
            'last_save_time': self.last_save_time
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PuzzleManager':
        """Deserialize from dictionary."""
        manager = cls()
        manager.total_score = data.get('total_score', 0.0)
        manager.creation_time = data.get('creation_time', time.time())
        manager.last_save_time = data.get('last_save_time', time.time())
        
        for n_str, puzzle_data in data.get('puzzles', {}).items():
            n = int(n_str)
            puzzle = PuzzleState.from_dict(puzzle_data)
            manager.puzzles[n] = puzzle
        
        return manager
    
    def mark_saved(self):
        """Update last save timestamp."""
        self.last_save_time = time.time()
