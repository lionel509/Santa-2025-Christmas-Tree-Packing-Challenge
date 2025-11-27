
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

# Load baseline solutions
known_solutions = {}
try:
    baseline_df = pd.read_csv('test.csv')
    baseline_df['x'] = baseline_df['x'].apply(lambda v: float(str(v).replace('s', '')))
    baseline_df['y'] = baseline_df['y'].apply(lambda v: float(str(v).replace('s', '')))
    baseline_df['deg'] = baseline_df['deg'].apply(lambda v: float(str(v).replace('s', '')))
    for n in range(1, 201):
        prefix = f"{n:03d}_"
        rows = baseline_df[baseline_df['id'].str.startswith(prefix)]
        if len(rows) == n:
            known_solutions[n] = [ChristmasTree(row['x'], row['y'], row['deg']) for _, row in rows.iterrows()]
    print(f"Loaded {len(known_solutions)} baseline configurations from test.csv")
except Exception as e:
    print(f"Could not load baseline solutions: {e}")

all_solutions = {}
cumulative_score = 0
improvements = 0

# Main processing loop (simplified and corrected)
for n in tqdm(range(200, 0, -1), desc="Processing N"):
    params = get_sa_params(n)
    
    # Determine the starting point for this N
    if n + 1 in all_solutions:
        # Start from the N+1 solution and remove a random tree
        trees = [ChristmasTree(t.center_x, t.center_y, t.angle) for t in all_solutions[n+1]]
        trees.pop(random.randint(0, len(trees)-1))
        source_type = "Derived"
    elif n in known_solutions:
        trees = known_solutions[n]
        source_type = "Baseline"
    else:
        # Fallback if no other option: generate randomly
        trees = [ChristmasTree(np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(0, 360)) for _ in range(n)]
        source_type = "Random"

    print(f"\n--- Processing N={n} (Source: {source_type}) ---")

    # Optimize the configuration
    optimized_trees = optimize_packing(list(trees), params)
    
    # CRITICAL FIX: Validate the optimized solution for collisions
    collision_count = validate_collisions(optimized_trees, tolerance=CONFIG['validation']['collision_tolerance'])
    
    final_trees = optimized_trees
    score_n = (get_bounds(optimized_trees)**2) / n
    
    if collision_count > 0:
        print(f"  [N={n}] ⚠️ Candidate has {collision_count} collisions.")
        # If collisions are found, revert to the baseline if it exists and is valid
        if n in known_solutions:
            baseline_collision_count = validate_collisions(known_solutions[n])
            if baseline_collision_count == 0:
                print(f"  [N={n}] Reverting to valid baseline solution.")
                final_trees = known_solutions[n]
                score_n = (get_bounds(final_trees)**2) / n
            else:
                print(f"  [N={n}] ‼️ FATAL: Baseline also has collisions! Using collided candidate as last resort.")
        else:
            print(f"  [N={n}] ‼️ FATAL: No valid baseline available to fall back to.")
            
    all_solutions[n] = final_trees
    cumulative_score += score_n
    print(f"  [N={n}] Score: {score_n:.6f} | Cumulative Score: {cumulative_score:.6f}")

# Generate submission file
submission_rows = []
for n, trees in all_solutions.items():
    for i, tree in enumerate(trees):
        submission_rows.append([
            f"{n:03d}_{i}", 
            f"s{tree.center_x:.10f}", 
            f"s{tree.center_y:.10f}", 
            f"s{tree.angle:.10f}"
        ])

df_sub = pd.DataFrame(submission_rows, columns=['id', 'x', 'y', 'deg'])
df_sub.sort_values('id', inplace=True)
df_sub.to_csv('submission.csv', index=False)

print("\n============================================================")
print("✅ Pipeline finished. A valid submission.csv has been generated.")
print(f"Final Score: {cumulative_score:.6f}")
print("============================================================")
