import pandas as pd
import numpy as np
import time

# ==========================================
# 1. Geometric Configuration & SAT Logic
# ==========================================

# Fixed Tree Polygon (0,0) at 0 degrees
BASE_POLY = np.array([
    (0, 0.8), (0.25, 0.5), (0.15, 0.5), (0.4, 0.25), (0.15, 0.25),
    (0.7, 0.0), (0.15, 0.0), (0.15, -0.2), (-0.15, -0.2), (-0.15, 0.0),
    (-0.7, 0.0), (-0.15, 0.25), (-0.4, 0.25), (-0.15, 0.5), (-0.25, 0.5)
])

def get_rotated_poly(x, y, deg):
    """Returns the vertices of the tree translated and rotated."""
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array(((c, -s), (s, c)))
    # Rotate
    rotated = BASE_POLY.dot(R.T)
    # Translate
    return rotated + np.array([x, y])

def get_axes(poly):
    """Get normals (axes) for SAT collision detection."""
    n = len(poly)
    axes = []
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        length = np.linalg.norm(normal)
        if length > 1e-9:
            axes.append(normal / length)
    return axes

def project(poly, axis):
    """Project polygon onto an axis."""
    dots = poly.dot(axis)
    return np.min(dots), np.max(dots)

def check_collision(poly_a, poly_b):
    """
    Checks if two polygons collide using Separating Axis Theorem (SAT).
    Returns True if they overlap. Returns False if they are separate or touching.
    """
    # 1. Bounding Box Check (Fast fail)
    min_a, max_a = np.min(poly_a, axis=0), np.max(poly_a, axis=0)
    min_b, max_b = np.min(poly_b, axis=0), np.max(poly_b, axis=0)

    # Use strict inequality with small epsilon for separation check
    if (max_a[0] < min_b[0] - 1e-7 or min_a[0] > max_b[0] + 1e-7 or
        max_a[1] < min_b[1] - 1e-7 or min_a[1] > max_b[1] + 1e-7):
        return False

    # 2. SAT Check (Precise)
    axes = get_axes(poly_a) + get_axes(poly_b)
    
    for axis in axes:
        min_p1, max_p1 = project(poly_a, axis)
        min_p2, max_p2 = project(poly_b, axis)
        
        # Calculate separation
        # If there is a gap > 0, they separate. 
        # We allow gap >= -epsilon (essentially touching or separate).
        if max_p1 < min_p2 - 1e-7 or max_p2 < min_p1 - 1e-7:
            return False # Separation found
            
    return True # Overlap detected

def is_valid_state(polygons, active_indices, static_indices):
    """
    Checks if specific active polygons collide with any static polygons.
    """
    for i in active_indices:
        poly1 = polygons[i]
        for j in static_indices:
            if i == j: continue 
            if check_collision(poly1, polygons[j]):
                return False
    return True

# ==========================================
# 2. Optimization Routines
# ==========================================

def solve_standard_squeeze(puzzle_id, df_puzzle):
    """
    Squeeze logic for standard puzzles. 
    Identifies edge trees and nudges them inward.
    """
    n = len(df_puzzle)
    coords = df_puzzle[['x', 'y', 'deg']].values.astype(float)
    polys = [get_rotated_poly(c[0], c[1], c[2]) for c in coords]
    
    # Calculate Initial Score
    all_points = np.vstack(polys)
    min_x, max_x = np.min(all_points[:,0]), np.max(all_points[:,0])
    min_y, max_y = np.min(all_points[:,1]), np.max(all_points[:,1])
    initial_s = max(max_x - min_x, max_y - min_y)
    
    # Step sizes to try (coarse to fine)
    steps = [0.1, 0.01, 0.001]
    
    for step in steps:
        improved_in_step = True
        while improved_in_step:
            improved_in_step = False
            
            # Re-calculate bounds
            all_points = np.vstack(polys)
            min_x, max_x = np.min(all_points[:,0]), np.max(all_points[:,0])
            min_y, max_y = np.min(all_points[:,1]), np.max(all_points[:,1])
            w, h = max_x - min_x, max_y - min_y
            
            threshold = 0.001
            moves = []
            
            # Squeeze dominant dimension, or both if square-ish
            if w >= h - 1e-5:
                left_idxs = [i for i, p in enumerate(polys) if np.min(p[:,0]) <= min_x + threshold]
                right_idxs = [i for i, p in enumerate(polys) if np.max(p[:,0]) >= max_x - threshold]
                moves.append((step, 0, left_idxs))   # Left -> Right
                moves.append((-step, 0, right_idxs)) # Right -> Left
            
            if h >= w - 1e-5:
                bottom_idxs = [i for i, p in enumerate(polys) if np.min(p[:,1]) <= min_y + threshold]
                top_idxs = [i for i, p in enumerate(polys) if np.max(p[:,1]) >= max_y - threshold]
                moves.append((0, step, bottom_idxs)) # Bottom -> Up
                moves.append((0, -step, top_idxs))   # Top -> Down
            
            for dx, dy, indices in moves:
                for idx in indices:
                    original_poly = polys[idx].copy()
                    
                    # Apply move
                    polys[idx][:, 0] += dx
                    polys[idx][:, 1] += dy
                    
                    # Check validity
                    indices_to_check = [x for x in range(n) if x != idx]
                    
                    if is_valid_state(polys, [idx], indices_to_check):
                        coords[idx, 0] += dx
                        coords[idx, 1] += dy
                        improved_in_step = True
                    else:
                        # Revert if invalid
                        polys[idx] = original_poly

    # Final Score
    final_points = np.vstack(polys)
    f_min_x, f_max_x = np.min(final_points[:,0]), np.max(final_points[:,0])
    f_min_y, f_max_y = np.min(final_points[:,1]), np.max(final_points[:,1])
    final_s = max(f_max_x - f_min_x, f_max_y - f_min_y)
    
    df_puzzle['x'] = coords[:, 0]
    df_puzzle['y'] = coords[:, 1]
    
    return df_puzzle, initial_s, final_s

def solve_grid_squeeze(puzzle_id, df_puzzle):
    """
    Column-based squeezing for grid puzzles (161-184).
    """
    n = len(df_puzzle)
    coords = df_puzzle[['x', 'y', 'deg']].values.astype(float)
    polys = [get_rotated_poly(c[0], c[1], c[2]) for c in coords]
    
    all_points = np.vstack(polys)
    w = np.max(all_points[:,0]) - np.min(all_points[:,0])
    h = np.max(all_points[:,1]) - np.min(all_points[:,1])
    initial_s = max(w, h)
    
    # Group by X-coordinate
    df_puzzle['col_group'] = df_puzzle['x'].apply(lambda x: round(x, 1))
    unique_cols = sorted(df_puzzle['col_group'].unique())
    mid_x = (np.max(all_points[:,0]) + np.min(all_points[:,0])) / 2
    
    steps = [0.1, 0.01, 0.001]
    
    for step in steps:
        improved = True
        while improved:
            improved = False
            
            for col_val in unique_cols:
                # Find trees in this column
                col_indices = [i for i in range(n) if abs(coords[i,0] - col_val) < 0.2]
                
                # Determine move direction towards center
                if col_val < mid_x - 0.1:
                    dx = step
                elif col_val > mid_x + 0.1:
                    dx = -step
                else:
                    continue
                
                saved_polys = [polys[i].copy() for i in col_indices]
                
                # Move entire column
                for i in col_indices:
                    polys[i][:, 0] += dx
                
                others = [x for x in range(n) if x not in col_indices]
                
                if is_valid_state(polys, col_indices, others):
                    for i in col_indices: coords[i, 0] += dx
                    improved = True
                else:
                    # Revert
                    for k, idx in enumerate(col_indices):
                        polys[idx] = saved_polys[k]
                        
    final_points = np.vstack(polys)
    fw = np.max(final_points[:,0]) - np.min(final_points[:,0])
    fh = np.max(final_points[:,1]) - np.min(final_points[:,1])
    final_s = max(fw, fh)
    
    df_puzzle['x'] = coords[:, 0]
    df_puzzle['y'] = coords[:, 1]
    if 'col_group' in df_puzzle.columns:
        del df_puzzle['col_group']
        
    return df_puzzle, initial_s, final_s

# ==========================================
# 3. Main Execution
# ==========================================

def main():
    print("Reading Data...")
    df = pd.read_csv('ml_packing_system/data/test.csv')
    
    # Clean 's' prefix
    for c in ['x', 'y', 'deg']:
        df[c] = df[c].astype(str).str.replace('s', '').astype(float)

    print("Data Loaded. Starting Optimization...")
    
    optimized_dfs = []
    total_start = time.time()
    
    puzzle_ids = df['id'].apply(lambda x: x.split('_')[0]).unique()

    for pid in puzzle_ids:
        mask = df['id'].str.startswith(f"{pid}_")
        df_puzzle = df[mask].copy().reset_index(drop=True)
        puzzle_num = int(pid)
        
        # Logic Selection
        if 161 <= puzzle_num <= 184:
            df_opt, s_start, s_end = solve_grid_squeeze(puzzle_num, df_puzzle)
            strat = "Grid"
        else:
            df_opt, s_start, s_end = solve_standard_squeeze(puzzle_num, df_puzzle)
            strat = "Std"
            
        imp = s_start - s_end
        print(f"Puzzle {pid} ({strat}): {s_start:.4f} -> {s_end:.4f} | Imp: {imp:.4f}")
        
        df_opt['id'] = df.loc[mask, 'id'].values
        optimized_dfs.append(df_opt)

    print(f"Optimization Complete. Time: {time.time() - total_start:.2f}s")
    
    final_df = pd.concat(optimized_dfs)
    
    # Re-add 's' prefix for submission format requirements
    for c in ['x', 'y', 'deg']:
        final_df[c] = 's' + final_df[c].astype(str)
        
    final_df.to_csv('submission_optimized.csv', index=False)
    print("Saved to submission_optimized.csv")

if __name__ == "__main__":
    main()