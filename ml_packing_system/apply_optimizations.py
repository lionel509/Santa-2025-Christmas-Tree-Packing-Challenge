
import sys
import os
import math
import numpy as np
from pathlib import Path

# Add the current directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.state.storage import LayoutStorage
from app.geometry.collision import check_all_collisions, check_collision
from app.geometry.tree import ChristmasTree

def squeeze_axis(puzzle, axis='x', step=0.01, max_steps=100):
    """
    Squeezes the puzzle along the specified axis by moving outer trees inward.
    """
    trees = puzzle.trees
    
    # Determine axis index
    axis_idx = 0 if axis == 'x' else 1
    
    # Sort trees by position on axis
    # We want to move leftmost/bottommost trees UP/RIGHT
    # And rightmost/topmost trees DOWN/LEFT
    
    # Identify bounds
    min_val = float('inf')
    max_val = float('-inf')
    
    for tree in trees:
        bounds = tree.get_bounds()
        min_val = min(min_val, bounds[axis_idx])
        max_val = max(max_val, bounds[axis_idx + 2])
    
    center = (min_val + max_val) / 2.0
    
    moved_count = 0
    
    for i, tree in enumerate(trees):
        # Determine direction towards center
        pos = tree.x if axis == 'x' else tree.y
        
        if pos < center:
            direction = 1.0 # Move right/up
        else:
            direction = -1.0 # Move left/down
            
        # Try to move
        original_pos = pos
        best_pos = pos
        
        for _ in range(max_steps):
            new_pos = best_pos + (direction * step)
            
            # Update tree
            if axis == 'x':
                tree.x = new_pos
            else:
                tree.y = new_pos
            
            # Check collisions
            # We only need to check this tree against others
            collision = False
            for j, other in enumerate(trees):
                if i == j:
                    continue
                if check_collision(tree, other):
                    collision = True
                    break
            
            if collision:
                # Revert to last valid
                if axis == 'x':
                    tree.x = best_pos
                else:
                    tree.y = best_pos
                break
            else:
                best_pos = new_pos
        
        if abs(best_pos - original_pos) > 1e-6:
            moved_count += 1
            
    return moved_count

def apply_specific_changes(manager):
    """Applies specific changes based on the user's analysis."""
    
    print("Applying specific optimizations...")
    
    for n in range(1, 201):
        puzzle = manager.get_puzzle(n)
        if not puzzle:
            continue
            
        trees = puzzle.trees
        
        # --- Specific Instructions ---
        
        if n == 1:
            # Rotate tree 001_0 to 10 deg
            print(f"Puzzle {n}: Rotating tree 0 to 10 deg")
            trees[0].deg = 10.0
            # Check collision
            if check_all_collisions(trees):
                print(f"Puzzle {n}: Collision at 10 deg, trying -10")
                trees[0].deg = -10.0
        
        elif n == 2:
            # Move tree 002_1 closer to center
            print(f"Puzzle {n}: Moving tree 1 closer to center")
            # Simple approach: move towards (0,0)
            t = trees[1]
            t.x *= 0.9
            t.y *= 0.9
            
        elif n == 3:
            # Shift tree 003_2 down or 003_1 up
            print(f"Puzzle {n}: Shifting tree 2 down")
            trees[2].y -= 0.1
            
        elif n == 5:
            # Shift 005_1, 005_2 down
            print(f"Puzzle {n}: Shifting trees 1, 2 down")
            trees[1].y -= 0.1
            trees[2].y -= 0.1
            
        elif n == 8:
            # Rotate entire cluster by 45
            print(f"Puzzle {n}: Rotating cluster by 45 deg")
            rad = np.deg2rad(45)
            cos_a = np.cos(rad)
            sin_a = np.sin(rad)
            for t in trees:
                nx = t.x * cos_a - t.y * sin_a
                ny = t.x * sin_a + t.y * cos_a
                t.x = nx
                t.y = ny
                t.deg += 45
                
        elif n == 11:
            # Shift 011_3 up and 011_9 down
            print(f"Puzzle {n}: Shifting 3 up, 9 down")
            if len(trees) > 3: trees[3].y += 0.1
            if len(trees) > 9: trees[9].y -= 0.1
            
        elif n == 18:
            # Move 018_3, 018_6 down
            print(f"Puzzle {n}: Moving 3, 6 down")
            if len(trees) > 3: trees[3].y -= 0.1
            if len(trees) > 6: trees[6].y -= 0.1
            
        elif n == 21:
            # Rotate 90?
            print(f"Puzzle {n}: Rotating cluster by 90 deg")
            for t in trees:
                nx = -t.y
                ny = t.x
                t.x = nx
                t.y = ny
                t.deg += 90

        elif n == 91:
             # Rotate 45?
            print(f"Puzzle {n}: Rotating cluster by 45 deg")
            rad = np.deg2rad(45)
            cos_a = np.cos(rad)
            sin_a = np.sin(rad)
            for t in trees:
                nx = t.x * cos_a - t.y * sin_a
                ny = t.x * sin_a + t.y * cos_a
                t.x = nx
                t.y = ny
                t.deg += 45
        
        # --- General Strategies ---
        
        # Compress Y (Height limited)
        compress_y_puzzles = [
            4, 5, 9, 10, 11, 12, 17, 18, 19, 20, 25, 26, 27, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40,
            45, 49, 50, 51, 52, 53, 54, 55, 56, 71, 72, 73, 74, 75, 82, 83, 84, 85, 86, 87, 88, 89, 90,
            103, 104, 105, 106, 107, 113, 114, 115, 116, 117, 118, 119, 120, 134, 135, 136, 137, 138, 139,
            140, 141, 142, 143, 144, 145, 146, 147, 153, 154, 155, 156, 157, 158, 159, 160,
            193, 194, 195, 196, 197, 198, 199, 200
        ]
        
        # Compress X (Width limited)
        compress_x_puzzles = [
            6, 7, 13, 15, 16, 22, 23, 24, 28, 34, 41, 42, 43, 44, 46, 47, 48, 57, 58, 59, 60, 61, 62, 63, 64,
            65, 66, 67, 68, 69, 70, 76, 77, 78, 79, 80, 81, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
            108, 109, 110, 111, 112, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
            148, 149, 150, 151, 152, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174,
            175, 176, 177, 178, 179, 180, 181, 182, 183, 184
        ]
        
        if n in compress_y_puzzles:
            print(f"Puzzle {n}: Compressing Y")
            squeeze_axis(puzzle, axis='y', step=0.05, max_steps=50)
            squeeze_axis(puzzle, axis='y', step=0.01, max_steps=50)
            
        if n in compress_x_puzzles:
            print(f"Puzzle {n}: Compressing X")
            squeeze_axis(puzzle, axis='x', step=0.05, max_steps=50)
            squeeze_axis(puzzle, axis='x', step=0.01, max_steps=50)
            
        # Update metrics
        puzzle.update_metrics()
        
        # Check if valid
        if check_all_collisions(trees):
            print(f"Puzzle {n}: WARNING - Collisions detected after optimization!")
            # In a real scenario, we might want to revert. 
            # For now, we'll leave it and let the user know.

def main():
    storage = LayoutStorage("ml_packing_system/data")
    
    # Load latest submission
    input_file = "ml_packing_system/data/test.csv"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
        
    print(f"Loading {input_file}...")
    manager = storage.import_csv(input_file)
    
    if not manager:
        print("Failed to load manager.")
        return
        
    # Apply optimizations
    apply_specific_changes(manager)
    
    # Save result
    print("Exporting new submission...")
    storage.export_submission(manager)
    print("Done!")

if __name__ == "__main__":
    main()
