"""Persistent storage for puzzle layouts."""

import json
import os
import time
from pathlib import Path
from typing import Optional
from .puzzle import PuzzleManager


class LayoutStorage:
    """Handles saving and loading puzzle states to/from disk."""
    
    def __init__(self, storage_dir: str = "data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.storage_dir / "puzzle_states.json"
        self.backup_file = self.storage_dir / "puzzle_states_backup.json"
        
    def save(self, manager: PuzzleManager):
        """
        Save puzzle manager state to disk.
        
        Args:
            manager: PuzzleManager instance to save
        """
        data = manager.to_dict()
        
        # Create backup of existing file
        if self.state_file.exists():
            try:
                self.state_file.replace(self.backup_file)
            except Exception:
                pass  # Backup failed, continue anyway
        
        # Write new state
        try:
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
            manager.mark_saved()
            return True
        except Exception as e:
            print(f"Error saving state: {e}")
            return False
    
    def load(self) -> Optional[PuzzleManager]:
        """
        Load puzzle manager state from disk.
        
        Returns:
            PuzzleManager instance or None if no saved state exists
        """
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
            return PuzzleManager.from_dict(data)
        except Exception as e:
            print(f"Error loading state: {e}")
            
            # Try backup
            if self.backup_file.exists():
                try:
                    with open(self.backup_file, 'r') as f:
                        data = json.load(f)
                    print("Loaded from backup file")
                    return PuzzleManager.from_dict(data)
                except Exception as e2:
                    print(f"Error loading backup: {e2}")
            
            return None
    
    def auto_save_loop(self, manager: PuzzleManager, interval_seconds: int = 300):
        """
        Periodically save state (run in background thread).
        
        Args:
            manager: PuzzleManager to save
            interval_seconds: Save interval
        """
        import threading
        
        def save_periodically():
            while True:
                time.sleep(interval_seconds)
                self.save(manager)
                print(f"Auto-saved at {time.strftime('%H:%M:%S')}")
        
        thread = threading.Thread(target=save_periodically, daemon=True)
        thread.start()
        return thread
    
    def export_submission(self, manager: PuzzleManager, output_file: Optional[str] = None) -> bool:
        """
        Export submission CSV file with detailed filename and metrics.
        
        Args:
            manager: PuzzleManager with puzzle states
            output_file: Optional output CSV filename (if None, auto-generates detailed name)
        
        Returns:
            True if successful
        """
        from datetime import datetime
        
        try:
            # Calculate metrics
            total_score = manager.total_score
            target_score = 70.0
            baseline_score = 201.81
            improvement_from_baseline = baseline_score - total_score
            remaining_to_target = total_score - target_score
            accuracy_pct = max(0, min(100, (improvement_from_baseline / (baseline_score - target_score)) * 100))
            
            # Generate detailed filename if not provided
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if output_file is None:
                filename = f'submission_score{total_score:.2f}_acc{accuracy_pct:.1f}pct_improved{improvement_from_baseline:.2f}_{timestamp}.csv'
                output_path = self.storage_dir / filename
            else:
                output_path = self.storage_dir / output_file
            
            # Also create a "latest" copy
            latest_path = self.storage_dir / 'submission_latest.csv'
            
            with open(output_path, 'w') as f:
                f.write("id,x,y,deg\n")
                
                for n in range(1, 201):
                    puzzle = manager.get_puzzle(n)
                    if puzzle is None:
                        print(f"Warning: Missing puzzle for n={n}")
                        continue
                    
                    for idx, tree in enumerate(puzzle.trees):
                        tree_id = f"{n:03d}_{idx}"
                        x_str = f"s{tree.x:.6f}"
                        y_str = f"s{tree.y:.6f}"
                        deg_str = f"s{tree.deg:.6f}"
                        f.write(f"{tree_id},{x_str},{y_str},{deg_str}\n")
            
            # Copy to latest
            import shutil
            shutil.copy2(output_path, latest_path)
            
            # Write detailed metrics file
            metrics_path = self.storage_dir / f'{output_path.stem}_metrics.txt'
            with open(metrics_path, 'w') as f:
                f.write(f"Santa 2025 - Christmas Tree Packing Challenge\n")
                f.write(f"{'='*70}\n\n")
                f.write(f"Submission Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Submission File: {output_path.name}\n\n")
                f.write(f"{'='*70}\n")
                f.write(f"SCORE METRICS\n")
                f.write(f"{'='*70}\n")
                f.write(f"Total Score:                 {total_score:.6f}\n")
                f.write(f"Target Score:                {target_score:.6f}\n")
                f.write(f"Baseline Score:              {baseline_score:.6f}\n")
                f.write(f"\nImprovement from Baseline:   {improvement_from_baseline:.6f} ({(improvement_from_baseline/baseline_score)*100:.2f}%)\n")
                f.write(f"Remaining to Target:         {remaining_to_target:.6f}\n")
                f.write(f"Progress to Target:          {accuracy_pct:.2f}%\n")
                f.write(f"\n{'='*70}\n")
                f.write(f"PUZZLE BREAKDOWN\n")
                f.write(f"{'='*70}\n")
                f.write(f"Total Puzzles:               200\n")
                f.write(f"Total Tree Placements:       20,100\n\n")
                
                # Top 10 best scores
                puzzle_scores = [(n, state.score) for n, state in manager.puzzles.items()]
                puzzle_scores.sort(key=lambda x: x[1])
                
                f.write(f"Top 10 Best Scores:\n")
                for i, (n, score) in enumerate(puzzle_scores[:10], 1):
                    f.write(f"  {i:2d}. Puzzle #{n:3d}: {score:.6f}\n")
                
                f.write(f"\nTop 10 Worst Scores:\n")
                for i, (n, score) in enumerate(puzzle_scores[-10:][::-1], 1):
                    f.write(f"  {i:2d}. Puzzle #{n:3d}: {score:.6f}\n")
                
                f.write(f"\n{'='*70}\n")
                f.write(f"OPTIMIZATION STATUS\n")
                f.write(f"{'='*70}\n")
                total_iters = sum(state.iterations for state in manager.puzzles.values())
                avg_iters = total_iters / len(manager.puzzles) if manager.puzzles else 0
                f.write(f"Total Iterations:            {total_iters:,}\n")
                f.write(f"Average Iterations/Puzzle:   {avg_iters:.1f}\n")
                f.write(f"\n{'='*70}\n")
            
            print(f"\n✅ Submission exported: {output_path.name}")
            print(f"📊 Score: {total_score:.6f} | Progress: {accuracy_pct:.1f}% to target")
            print(f"📈 Improved {improvement_from_baseline:.2f} from baseline ({(improvement_from_baseline/baseline_score)*100:.1f}%)")
            print(f"🔗 Latest copy: {latest_path.name}")
            print(f"📝 Metrics: {metrics_path.name}")
            
            return True
            
        except Exception as e:
            print(f"Error exporting submission: {e}")
            return False
