import sys
import os
import math
import numpy as np
import copy
from pathlib import Path

# Add the current directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.state.storage import LayoutStorage
from app.geometry.tree import ChristmasTree
from app.geometry.bounds import calculate_bounding_square, get_bounding_box

def center_trees(trees):
    """Center the cluster of trees at the origin."""
    if not trees:
        return trees
    
    # Calculate current bounds
    minx, miny, maxx, maxy = get_bounding_box(trees)
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    
    new_trees = []
    for tree in trees:
        t = tree.copy()
        t.x -= cx
        t.y -= cy
        new_trees.append(t)
    return new_trees

def rotate_trees(trees, angle_deg):
    """Rotate all trees by angle_deg around the origin."""
    rad = np.deg2rad(angle_deg)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    
    new_trees = []
    for tree in trees:
        # Rotate position
        nx = tree.x * cos_a - tree.y * sin_a
        ny = tree.x * sin_a + tree.y * cos_a
        
        # Rotate orientation
        ndeg = tree.deg + angle_deg
        
        new_trees.append(ChristmasTree(nx, ny, ndeg))
    return new_trees

def flip_trees(trees):
    """Flip trees across the X-axis (y -> -y)."""
    new_trees = []
    for tree in trees:
        # Flip position
        nx = tree.x
        ny = -tree.y
        
        # Flip orientation (negate angle)
        ndeg = -tree.deg
        
        new_trees.append(ChristmasTree(nx, ny, ndeg))
    return new_trees

def optimize_orientation(puzzle_id, trees):
    """Find the best rotation/flip for the given trees."""
    # First, center the trees to make rotation well-behaved
    centered_trees = center_trees(trees)
    
    best_trees = [t.copy() for t in centered_trees]
    best_side = calculate_bounding_square(best_trees)
    
    # Keep track of best transformation
    best_angle = 0.0
    best_flipped = False
    
    # 1. Rotate 360
    # 2. Flip
    # 3. Rotate 360 (of the flipped)
    
    # We can combine this: For flipped in [False, True], rotate 0..360
    
    # Coarse search step
    step = 5 # 5 degrees step for speed, then refine
    
    for flipped in [False, True]:
        base_set = centered_trees if not flipped else flip_trees(centered_trees)
        
        for angle in range(0, 360, step):
            rotated = rotate_trees(base_set, angle)
            side = calculate_bounding_square(rotated)
            
            if side < best_side - 1e-9:
                best_side = side
                best_trees = rotated
                best_angle = angle
                best_flipped = flipped
    
    # Refine search around best angle
    # Search +/- step with finer grain
    if best_angle is not None:
        start = best_angle - step
        end = best_angle + step
        
        base_set = centered_trees if not best_flipped else flip_trees(centered_trees)
        
        for angle in np.arange(start, end, 0.5):
            rotated = rotate_trees(base_set, angle)
            side = calculate_bounding_square(rotated)
            
            if side < best_side - 1e-9:
                best_side = side
                best_trees = rotated
                best_angle = angle
                
    # Final refinement
    if best_angle is not None:
        start = best_angle - 0.5
        end = best_angle + 0.5
        
        base_set = centered_trees if not best_flipped else flip_trees(centered_trees)
        
        for angle in np.arange(start, end, 0.1):
            rotated = rotate_trees(base_set, angle)
            side = calculate_bounding_square(rotated)
            
            if side < best_side - 1e-9:
                best_side = side
                best_trees = rotated
    
    improvement = calculate_bounding_square(trees) - best_side
    if improvement > 1e-6:
        print(f"Puzzle {puzzle_id}: Improved by {improvement:.6f} (Side: {best_side:.6f})")
        
    return best_trees

def main():
    storage = LayoutStorage("ml_packing_system/data")
    
    # Find latest submission file
    data_dir = Path("ml_packing_system/data")
    submission_files = list(data_dir.glob("submission_*.csv"))
    if not submission_files:
        print("No submission files found.")
        input_file = "ml_packing_system/data/test.csv"
    else:
        submission_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        input_file = str(submission_files[0])
    
    print(f"Loading {input_file}...")
    manager = storage.import_csv(input_file)
    
    if not manager:
        print("Failed to load manager.")
        return
        
    total_improvement = 0.0
    
    print("Optimizing orientations...")
    for n in range(1, 201):
        puzzle = manager.get_puzzle(n)
        if not puzzle:
            continue
            
        original_score = puzzle.score
        original_side = puzzle.side_length
        
        optimized_trees = optimize_orientation(n, puzzle.trees)
        
        # Update puzzle
        puzzle.trees = optimized_trees
        puzzle.update_metrics()
        
        if puzzle.side_length < original_side - 1e-9:
            improvement = original_score - puzzle.score
            total_improvement += improvement
            manager.add_puzzle(puzzle)
        
    print(f"Total score improvement: {total_improvement:.4f}")
    
    print("Exporting new submission...")
    storage.export_submission(manager)
    print("Done!")

if __name__ == "__main__":
    main()
