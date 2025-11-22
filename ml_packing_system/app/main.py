"""Main application orchestrator."""

import asyncio
import threading
import time
import random
from typing import Optional, List

from app.state import PuzzleManager, LayoutStorage
from app.optimizers import initialize_all_puzzles, HybridOptimizer
from app.ml import PPOAgent
from app.api.websocket import ws_manager


class Application:
    """Main application orchestrator."""
    
    def __init__(
        self,
        use_ml: bool = True,
        device: str = 'cpu',
        auto_save_interval: int = 300
    ):
        """
        Initialize application.
        
        Args:
            use_ml: Whether to use ML optimization
            device: Device for ML computation
            auto_save_interval: Auto-save interval in seconds
        """
        self.use_ml = use_ml
        self.device = device
        self.auto_save_interval = auto_save_interval
        self.backpacking_mode = False
        self.skip_cycle_1 = False
        self.baseline_file: Optional[str] = None
        
        # Components
        self.manager: Optional[PuzzleManager] = None
        self.storage: Optional[LayoutStorage] = None
        self.optimizer: Optional[HybridOptimizer] = None
        self.agent: Optional[PPOAgent] = None
        
        # Threading
        self.optimization_threads: List[threading.Thread] = []
        self.running = False
        self.num_workers = 4  # Default number of parallel workers
    
    def initialize(self):
        """Initialize all components."""
        print("Initializing Santa 2025 ML Packing System...")
        
        # Storage
        self.storage = LayoutStorage("data")
        
        # Try to load existing state
        self.manager = self.storage.load()
        
        # If baseline file provided, load it (overwriting or initializing)
        if self.baseline_file:
            print(f"Loading baseline from {self.baseline_file}...")
            baseline_manager = self.storage.import_csv(self.baseline_file)
            if baseline_manager:
                if self.manager is None:
                    self.manager = baseline_manager
                    print("Initialized manager from baseline CSV")
                else:
                    # Merge baseline into existing manager
                    print("Merging baseline into existing state...")
                    count = 0
                    for n, puzzle in baseline_manager.puzzles.items():
                        current = self.manager.get_puzzle(n)
                        if current is None or puzzle.score < current.score:
                            self.manager.add_puzzle(puzzle)
                            count += 1
                    print(f"Merged {count} better solutions from baseline")
                
                # Save immediately
                self.storage.save(self.manager)
        
        if self.manager is None:
            print("No saved state found. Initializing all puzzles...")
            self.manager = PuzzleManager()
            
            # Initialize all puzzles
            puzzles = initialize_all_puzzles(method='greedy')
            for n, puzzle in puzzles.items():
                self.manager.add_puzzle(puzzle)
            
            # Save initial state
            self.storage.save(self.manager)
            print(f"Initialized {len(puzzles)} puzzles")
        else:
            print(f"Loaded {len(self.manager.puzzles)} puzzles from saved state")
        
        # Initialize ML agent if enabled
        if self.use_ml:
            print("Initializing ML agent...")
            try:
                import torch
                
                # State dimension: 200 trees * 3 + 4 global features
                state_dim = 200 * 3 + 4
                
                self.agent = PPOAgent(
                    state_dim=state_dim,
                    num_trees=200,  # Max trees
                    device=self.device
                )
                print("ML agent initialized")
            except Exception as e:
                print(f"Warning: Could not initialize ML agent: {e}")
                print("Continuing with heuristics only")
                self.use_ml = False
        
        # Initialize optimizer
        self.optimizer = HybridOptimizer(
            agent=self.agent,
            device=self.device,
            use_ml=self.use_ml,
            use_heuristics=True
        )
        
        print("Initialization complete!")
        print(f"Total score: {self.manager.total_score:.2f}")
        print(f"Average score: {self.manager.total_score / len(self.manager.puzzles):.6f}")
    
    def start_optimization(self):
        """Start optimization in background threads."""
        if self.running:
            print("Optimization already running")
            return
        
        self.running = True
        
        def optimization_loop(worker_id: int):
            """Main optimization loop - runs continuously 24/7."""
            # Seed randomness for this thread
            random.seed(time.time() + worker_id)
            
            print(f"\n[Worker {worker_id}] 🚀 STARTING OPTIMIZATION LOOP")
            
            cycle = 1 if self.skip_cycle_1 else 0
            # Track consecutive no-improvement trials per puzzle (local to worker)
            no_improvement_count = {n: 0 for n in range(1, 201)}
            max_trials_without_improvement = 50
            
            while self.running:
                cycle += 1
                cycle_start_time = time.time()
                cycle_improvements = 0
                puzzles_optimized = 0
                
                if worker_id == 0:
                    print(f"\n{'='*80}")
                    print(f"🔄 CYCLE {cycle} START - {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"{'='*80}")
                    print(f"📊 Current Status:")
                    summary = self.manager.get_summary()
                    print(f"   • Total Score: {summary['total_score']:.2f}")
                    print(f"   • Average Score: {summary['avg_score']:.6f}")
                    print(f"   • Total Iterations: {summary['total_iterations']:,}")
                    print()
                
                # Determine optimization order
                if self.backpacking_mode and cycle == 1:
                    order_desc = "FORWARD (1 → 200)"
                    puzzle_range = range(1, 201)
                else:
                    order_desc = "REVERSE (200 → 1)"
                    puzzle_range = range(200, 0, -1)

                if worker_id == 0:
                    print(f"🔄 Optimization Order: {order_desc}")
                    print()
                
                # Iterate through all 200 puzzles continuously
                for n in puzzle_range:
                    if not self.running:
                        break
                    
                    # Inner loop to stay on puzzle if improving (Backpacking mode Cycle 2+)
                    while True:
                        puzzle = self.manager.get_puzzle(n)
                        if puzzle is None:
                            if worker_id == 0:
                                print(f"⚠️  Puzzle {n:3d}: NOT FOUND - skipping")
                            break
                        
                        # --- BACKPACKING PROPAGATION (Cycle 2+, Reverse Order) ---
                        # Only Worker 0 does propagation to avoid race conditions/redundancy
                        if worker_id == 0 and self.backpacking_mode and cycle > 1 and n < 200:
                            parent_puzzle = self.manager.get_puzzle(n + 1)
                            if parent_puzzle is not None:
                                from app.state import PuzzleState
                                from app.optimizers.backpacking import smart_truncate
                                
                                candidate_trees = smart_truncate(parent_puzzle.trees, n)
                                candidate_state = PuzzleState(
                                    n=n,
                                    trees=candidate_trees,
                                    score=0.0,
                                    side_length=0.0,
                                    iterations=puzzle.iterations,
                                    collisions=0
                                )
                                candidate_state.update_metrics()
                                
                                if candidate_state.score < puzzle.score:
                                    improvement_diff = puzzle.score - candidate_state.score
                                    print(f"   🎁 BACKPACKING: Inherited better layout from {n+1}!")
                                    print(f"      {puzzle.score:.6f} → {candidate_state.score:.6f} (↓{improvement_diff:.6f})")
                                    self.manager.add_puzzle(candidate_state)
                                    puzzle = candidate_state
                                    no_improvement_count[n] = 0
                                    try:
                                        asyncio.run(ws_manager.broadcast_state_update(n, {
                                            'n': n,
                                            'score': candidate_state.score,
                                            'side_length': candidate_state.side_length,
                                            'iterations': candidate_state.iterations
                                        }))
                                    except:
                                        pass

                        # Skip if we've tried too many times without improvement
                        if no_improvement_count[n] >= max_trials_without_improvement:
                            break
                        
                        # Verbose progress every puzzle (only worker 0 logs details)
                        old_score = puzzle.score
                        puzzle_start_time = time.time()
                        
                        if worker_id == 0:
                            print(f"🌲 Puzzle {n:3d} ({n} trees) - Starting optimization...")
                            print(f"   • Current score: {old_score:.6f}")
                        
                        # Optimize with callback
                        iteration_improvements = [0]
                        
                        def callback(iteration, state):
                            if state.score < old_score:
                                iteration_improvements[0] += 1
                                try:
                                    asyncio.run(ws_manager.broadcast_improvement(n, old_score, state.score))
                                except:
                                    pass
                        
                        # Run optimization
                        optimized = self.optimizer.optimize_puzzle(
                            puzzle,
                            max_iterations=100,
                            callback=callback,
                            verbose=(worker_id == 0)
                        )
                        
                        # Update manager (thread-safe)
                        self.manager.add_puzzle(optimized)
                        puzzles_optimized += 1
                        
                        # Check for improvement
                        improvement = old_score - optimized.score
                        
                        should_repeat = False
                        if improvement > 1e-6:
                            no_improvement_count[n] = 0
                            cycle_improvements += 1
                            if worker_id == 0:
                                print(f"   ✅ IMPROVED: {old_score:.6f} → {optimized.score:.6f} (↓{improvement:.6f})")
                                # Broadcast global progress immediately on improvement
                                try:
                                    asyncio.run(ws_manager.broadcast_progress(self.manager.get_summary()))
                                except:
                                    pass
                            
                            if self.backpacking_mode and cycle > 1:
                                if worker_id == 0:
                                    print(f"   🔄 REPEATING: Staying on puzzle {n} due to improvement...")
                                should_repeat = True
                        else:
                            no_improvement_count[n] += 1
                            if worker_id == 0:
                                if no_improvement_count[n] >= max_trials_without_improvement:
                                    print(f"   ⏸️  PAUSED: No improvement after {max_trials_without_improvement} trials")
                                else:
                                    print(f"   ➡️  No improvement this trial ({no_improvement_count[n]}/{max_trials_without_improvement})")
                        
                        try:
                            asyncio.run(ws_manager.broadcast_state_update(n, {
                                'n': n,
                                'score': optimized.score,
                                'side_length': optimized.side_length,
                                'iterations': optimized.iterations
                            }))
                        except:
                            pass
                        
                        if worker_id == 0:
                            print()
                        
                        if not should_repeat or not self.running:
                            break
                
                # Cycle summary (Worker 0 only)
                if worker_id == 0:
                    cycle_time = time.time() - cycle_start_time
                    summary = self.manager.get_summary()
                    active_puzzles = sum(1 for count in no_improvement_count.values() if count < max_trials_without_improvement)
                    
                    print(f"\n{'='*80}")
                    print(f"📈 CYCLE {cycle} COMPLETE - {time.strftime('%H:%M:%S')}")
                    print(f"{'='*80}")
                    print(f"⏱️  Cycle Statistics:")
                    print(f"   • Cycle Duration: {cycle_time:.1f} seconds")
                    print(f"   • Improvements Found: {cycle_improvements}")
                    print()
                    
                    if active_puzzles == 0:
                        print("\n" + "🔄"*30)
                        print("🔄 ALL PUZZLES PAUSED - STARTING NEW OPTIMIZATION ROUND!")
                        print("🔄"*30)
                        no_improvement_count = {n: 0 for n in range(1, 201)}
                    
                    try:
                        asyncio.run(ws_manager.broadcast_progress(summary))
                    except:
                        pass
                    
                    if cycle % 3 == 0:
                        print(f"\n💾 AUTO-SAVE TRIGGERED (Cycle {cycle})")
                        self.storage.save(self.manager)
                    
                    print(f"\n⏭️  Moving to next cycle in 2 seconds...")
                    time.sleep(2)
        
        # Start workers
        print(f"Starting {self.num_workers} optimization workers...")
        for i in range(self.num_workers):
            t = threading.Thread(target=optimization_loop, args=(i,), daemon=True)
            self.optimization_threads.append(t)
            t.start()
        
        # Start auto-save thread
        self._start_auto_save()
    
    def _start_auto_save(self):
        """Start auto-save thread."""
        def auto_save_loop():
            save_count = 0
            while self.running:
                time.sleep(self.auto_save_interval)
                if self.running:
                    save_count += 1
                    print(f"\n[{time.strftime('%H:%M:%S')}] Auto-saving...")
                    self.storage.save(self.manager)
                    
                    # Export submission CSV every 3 saves (every 15 minutes by default)
                    if save_count % 3 == 0:
                        print(f"[{time.strftime('%H:%M:%S')}] Exporting submission CSV...")
                        self.storage.export_submission(self.manager)
        
        thread = threading.Thread(target=auto_save_loop, daemon=True)
        thread.start()
    
    def stop(self):
        """Stop optimization."""
        print("\nStopping optimization...")
        self.running = False
        
        for t in self.optimization_threads:
            t.join(timeout=5)
        
        # Final save
        if self.storage and self.manager:
            print("Saving final state...")
            self.storage.save(self.manager)
        
        print("Stopped")
    
    def export_submission(self, filename: str = "submission.csv"):
        """Export submission file."""
        if self.storage and self.manager:
            return self.storage.export_submission(self.manager, filename)
        return False


# Global application instance
app_instance: Optional[Application] = None


def get_app(initialize: bool = True) -> Application:
    """Get or create application instance."""
    global app_instance
    if app_instance is None:
        app_instance = Application()
        if initialize:
            app_instance.initialize()
    return app_instance
