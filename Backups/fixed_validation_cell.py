
def validate_no_collisions(trees, verbose=False, tolerance=1e-9):
    """
    Validates that no trees overlap using a spatial index (STRtree).
    Returns (is_valid, collision_count).
    """
    n = len(trees)
    if n <= 1:
        return True, 0

    polygons = [t.polygon for t in trees]
    tree_index = STRtree(polygons)
    
    collision_count = 0
    checked_pairs = set()

    for i, poly in enumerate(polygons):
        # Query for potential intersections
        query_indices = tree_index.query(poly)
        for j in query_indices:
            # Avoid self-comparison and duplicate pairs
            if i >= j:
                continue
            
            pair = tuple(sorted((i, j)))
            if pair in checked_pairs:
                continue
            
            checked_pairs.add(pair)
            
            p1 = polygons[pair[0]]
            p2 = polygons[pair[1]]

            # Check for actual intersection, ignoring simple touches
            if p1.intersects(p2) and not p1.touches(p2):
                intersection_area = p1.intersection(p2).area
                if intersection_area > tolerance:
                    collision_count += 1
                    if verbose:
                        print(f"   - Trees {pair[0]} and {pair[1]} overlap (area={intersection_area:.4e})")
            
    return collision_count == 0, collision_count
