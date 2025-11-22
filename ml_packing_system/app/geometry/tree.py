"""Christmas tree polygon definition and transformation."""

import numpy as np
from typing import Tuple, List


class ChristmasTree:
    """Represents a Christmas tree with position, rotation, and polygon geometry."""
    
    # Tree dimensions (fixed)
    TRUNK_W = 0.15
    TRUNK_H = 0.2
    BASE_W = 0.7
    MID_W = 0.4
    TOP_W = 0.25
    TIP_Y = 0.8
    TIER_1_Y = 0.5
    TIER_2_Y = 0.25
    BASE_Y = 0.0
    
    # Base polygon coordinates (centered at origin, 0-degree rotation)
    _BASE_POLYGON = None
    
    @classmethod
    def get_base_polygon(cls) -> np.ndarray:
        """Get the base tree polygon coordinates."""
        if cls._BASE_POLYGON is None:
            trunk_bottom_y = -cls.TRUNK_H
            cls._BASE_POLYGON = np.array([
                # Tip
                [0.0, cls.TIP_Y],
                # Right side - Top Tier
                [cls.TOP_W / 2, cls.TIER_1_Y],
                [cls.TOP_W / 4, cls.TIER_1_Y],
                # Right side - Middle Tier
                [cls.MID_W / 2, cls.TIER_2_Y],
                [cls.MID_W / 4, cls.TIER_2_Y],
                # Right side - Bottom Tier
                [cls.BASE_W / 2, cls.BASE_Y],
                # Right Trunk
                [cls.TRUNK_W / 2, cls.BASE_Y],
                [cls.TRUNK_W / 2, trunk_bottom_y],
                # Left Trunk
                [-cls.TRUNK_W / 2, trunk_bottom_y],
                [-cls.TRUNK_W / 2, cls.BASE_Y],
                # Left side - Bottom Tier
                [-cls.BASE_W / 2, cls.BASE_Y],
                # Left side - Middle Tier
                [-cls.MID_W / 4, cls.TIER_2_Y],
                [-cls.MID_W / 2, cls.TIER_2_Y],
                # Left side - Top Tier
                [-cls.TOP_W / 4, cls.TIER_1_Y],
                [-cls.TOP_W / 2, cls.TIER_1_Y],
            ], dtype=np.float64)
        return cls._BASE_POLYGON
    
    def __init__(self, x: float = 0.0, y: float = 0.0, deg: float = 0.0):
        """
        Initialize a Christmas tree.
        
        Args:
            x: X-coordinate of tree center (top of trunk)
            y: Y-coordinate of tree center (top of trunk)
            deg: Rotation angle in degrees
        """
        self.x = float(x)
        self.y = float(y)
        self.deg = float(deg)
        self._polygon_cache = None
    
    @property
    def polygon(self) -> np.ndarray:
        """Get the transformed polygon coordinates (cached)."""
        if self._polygon_cache is None:
            self._polygon_cache = self._compute_polygon()
        return self._polygon_cache
    
    def _compute_polygon(self) -> np.ndarray:
        """Compute the transformed polygon coordinates."""
        base = self.get_base_polygon()
        
        # Rotation matrix
        rad = np.deg2rad(self.deg)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        # Rotate and translate
        rotated = base @ rotation_matrix.T
        translated = rotated + np.array([self.x, self.y])
        
        return translated
    
    def update_position(self, x: float, y: float, deg: float):
        """Update tree position and rotation, invalidating cache."""
        self.x = float(x)
        self.y = float(y)
        self.deg = float(deg)
        self._polygon_cache = None
    
    def move(self, dx: float, dy: float, ddeg: float = 0.0):
        """Move tree by delta values."""
        self.update_position(self.x + dx, self.y + dy, self.deg + ddeg)
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box (minx, miny, maxx, maxy)."""
        poly = self.polygon
        return (
            float(poly[:, 0].min()),
            float(poly[:, 1].min()),
            float(poly[:, 0].max()),
            float(poly[:, 1].max())
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'x': self.x,
            'y': self.y,
            'deg': self.deg
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ChristmasTree':
        """Create tree from dictionary."""
        return cls(
            x=data['x'],
            y=data['y'],
            deg=data['deg']
        )
    
    def copy(self) -> 'ChristmasTree':
        """Create a copy of this tree."""
        return ChristmasTree(self.x, self.y, self.deg)
    
    def __repr__(self) -> str:
        return f"ChristmasTree(x={self.x:.6f}, y={self.y:.6f}, deg={self.deg:.6f})"
