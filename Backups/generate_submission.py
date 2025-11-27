"""Generate submission.csv from puzzle_states.json"""

import json
import sys
from pathlib import Path

def generate_submission():
    # Find puzzle_states.json
    possible_paths = [
        Path("Data/puzzle_states.json"),
        Path("ml_packing_system/data/puzzle_states.json"),
        Path("data/puzzle_states.json")
    ]
    
    puzzle_states_path = None
    for path in possible_paths:
        if path.exists():
            puzzle_states_path = path
            break
    
    if not puzzle_states_path:
        print("Error: Could not find puzzle_states.json")
        print("Searched in:")
        for path in possible_paths:
            print(f"  - {path}")
        return
    
    print(f"Loading from: {puzzle_states_path}")
    
    # Load puzzle states
    with open(puzzle_states_path, 'r') as f:
        data = json.load(f)
    
    puzzles = data.get('puzzles', {})
    
    if not puzzles:
        print("Error: No puzzles found in puzzle_states.json")
        return
    
    # Create submission directory
    output_dir = Path("ml_packing_system/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "submission.csv"
    
    print(f"Generating submission to: {output_file}")
    
    # Write submission CSV
    with open(output_file, 'w') as f:
        f.write("id,x,y,deg\n")
        
        total_rows = 0
        missing = []
        
        for n in range(1, 201):
            n_str = str(n)
            
            if n_str not in puzzles:
                missing.append(n)
                print(f"Warning: Missing puzzle {n}")
                continue
            
            puzzle = puzzles[n_str]
            trees = puzzle.get('trees', [])
            
            if len(trees) != n:
                print(f"Warning: Puzzle {n} has {len(trees)} trees, expected {n}")
            
            for idx, tree in enumerate(trees):
                tree_id = f"{n:03d}_{idx}"
                x = tree['x']
                y = tree['y']
                deg = tree['deg']
                
                # Format with 's' prefix and 6 decimal places
                x_str = f"s{x:.6f}"
                y_str = f"s{y:.6f}"
                deg_str = f"s{deg:.6f}"
                
                f.write(f"{tree_id},{x_str},{y_str},{deg_str}\n")
                total_rows += 1
    
    print(f"\n✅ Submission created!")
    print(f"   Total rows: {total_rows}")
    print(f"   Expected: 20,100 (sum of 1 to 200)")
    print(f"   Location: {output_file}")
    
    if missing:
        print(f"\n⚠️  Missing puzzles: {missing}")
    
    if total_rows == 20100:
        print("\n🎉 Perfect! All 20,100 rows generated!")
    else:
        print(f"\n⚠️  Row count mismatch. Got {total_rows}, expected 20,100")
    
    # Calculate total score
    total_score = sum(puzzles[str(n)]['score'] for n in range(1, 201) if str(n) in puzzles)
    print(f"\n📊 Current total score: {total_score:.2f}")
    print(f"   Target score: 70.00")
    print(f"   Improvement needed: {total_score - 70:.2f}")

if __name__ == "__main__":
    generate_submission()
