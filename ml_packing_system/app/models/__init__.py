"""Data models."""

from dataclasses import dataclass
from typing import List


@dataclass
class TreeData:
    """Tree data for API responses."""
    x: float
    y: float
    deg: float
    id: str


@dataclass
class PuzzleResponse:
    """Puzzle response model."""
    n: int
    score: float
    side_length: float
    iterations: int
    trees: List[TreeData]


@dataclass
class SystemStatus:
    """System status model."""
    total_puzzles: int
    total_score: float
    avg_score: float
    total_iterations: int
    uptime_seconds: float
