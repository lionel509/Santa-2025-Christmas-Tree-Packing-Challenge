"""Main application orchestrator."""

import asyncio
import threading
import time
from typing import Optional

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
        
        # Components
        self.manager: Optional[PuzzleManager] = None
        self.storage: Optional[LayoutStorage] = None
        self.optimizer: Optional[HybridOptimizer] = None
        self.agent: Optional[PPOAgent] = None
        
        # Threading
        self.optimization_thread: Optional[threading.Thread] = None
        self.running = False
    
    def initialize(self):
        """Initialize all components."""
        print("Initializing Santa 2025 ML Packing System...")
        
        # Storage
        self.storage = LayoutStorage("data")
        
        # Try to load existing state
        self.manager = self.storage.load()
        
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
        """Start optimization in background thread."""
        if self.running:
            print("Optimization already running")
            return
        
        self.running = True
        
        def optimization_loop():
            """Main optimization loop - runs continuously 24/7."""
            print("\n" + "="*80)
            print("🚀 STARTING CONTINUOUS OPTIMIZATION LOOP (24/7)")
            print("="*80)
            print("📋 Configuration:")
            print(f"   • Total Puzzles: 200 (1 to 200 trees each)")
            print(f"   • Optimization Order: REVERSE (200 → 1) - Hardest first!")
            print(f"   • Iterations per puzzle: 100")
            print(f"   • Early stopping: 50 consecutive trials without improvement")
            print(f"   • ML Agent: {'ENABLED' if self.use_ml else 'DISABLED'}")
            print(f"   • Device: {self.device}")
            print(f"   • Collision tolerance: 0.0 (ABSOLUTE ZERO GAP - STRICT)")
            print(f"   • Auto-save: Every 3 cycles")
            print("="*80)
            print()
            
            cycle = 0
            # Track consecutive no-improvement trials per puzzle
            no_improvement_count = {n: 0 for n in range(1, 201)}
            max_trials_without_improvement = 50
            
            while self.running:
                cycle += 1
                cycle_start_time = time.time()
                cycle_improvements = 0
                puzzles_optimized = 0
                
                print(f"\n{'='*80}")
                print(f"🔄 CYCLE {cycle} START - {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*80}")
                print(f"📊 Current Status:")
                summary = self.manager.get_summary()
                print(f"   • Total Score: {summary['total_score']:.2f}")
                print(f"   • Average Score: {summary['avg_score']:.6f}")
                print(f"   • Total Iterations: {summary['total_iterations']:,}")
                print()
                print(f"🔄 Optimization Order: REVERSE (200 → 1)")
                print(f"   Starting with harder puzzles (more trees) first!")
                print()
                
                # Iterate through all 200 puzzles continuously - REVERSE ORDER (200 to 1)
                for n in range(200, 0, -1):  # Start at 200, go down to 1
                    if not self.running:
                        print("\n⛔ Optimization stopped by user")
                        break
                    
                    puzzle = self.manager.get_puzzle(n)
                    if puzzle is None:
                        print(f"⚠️  Puzzle {n:3d}: NOT FOUND - skipping")
                        continue
                    
                    # Skip if we've tried too many times without improvement
                    if no_improvement_count[n] >= max_trials_without_improvement:
                        continue
                    
                    # Verbose progress every puzzle
                    old_score = puzzle.score
                    puzzle_start_time = time.time()
                    
                    print(f"🌲 Puzzle {n:3d} ({n} trees) - Starting optimization...")
                    print(f"   • Current score: {old_score:.6f}")
                    print(f"   • Side length: {puzzle.side_length:.4f}")
                    print(f"   • No-improvement count: {no_improvement_count[n]}/{max_trials_without_improvement}")
                    print(f"   • Total trees in puzzle: {len(puzzle.trees)}")
                    print(f"   • Running 100 optimization iterations...")
                    print(f"   • Progress will be reported every 20 iterations")
                    print(f"   • ML Agent: {'ACTIVE' if self.use_ml else 'DISABLED'}")
                    print(f"   • Please wait... (this may take 10-30 seconds)")
                    print()
                    
                    # Optimize with callback
                    iteration_improvements = [0]  # Track improvements during iterations
                    
                    def callback(iteration, state):
                        if state.score < old_score:
                            iteration_improvements[0] += 1
                            # Broadcast improvement via WebSocket
                            try:
                                asyncio.run(
                                    ws_manager.broadcast_improvement(
                                        n, old_score, state.score
                                    )
                                )
                            except:
                                pass
                    
                    # Run optimization with more aggressive iterations
                    optimized = self.optimizer.optimize_puzzle(
                        puzzle,
                        max_iterations=100,
                        callback=callback,
                        verbose=True  # Enable verbose logging
                    )
                    
                    puzzle_time = time.time() - puzzle_start_time
                    print(f"\n   ⏱️  Optimization completed in {puzzle_time:.2f}s")
                    
                    # Update manager
                    self.manager.add_puzzle(optimized)
                    puzzles_optimized += 1
                    
                    # Check for improvement
                    improvement = old_score - optimized.score
                    
                    if improvement > 1e-6:  # Any improvement (even tiny)
                        no_improvement_count[n] = 0  # Reset counter
                        cycle_improvements += 1
                        print(f"   ✅ IMPROVED: {old_score:.6f} → {optimized.score:.6f} (↓{improvement:.6f})")
                        print(f"   • New side length: {optimized.side_length:.4f}")
                        print(f"   • Iteration improvements: {iteration_improvements[0]}")
                        print(f"   • Trial counter: RESET to 0")
                    else:
                        no_improvement_count[n] += 1
                        if no_improvement_count[n] >= max_trials_without_improvement:
                            print(f"   ⏸️  PAUSED: No improvement after {max_trials_without_improvement} trials")
                            print(f"   • Final score: {optimized.score:.6f}")
                        else:
                            print(f"   ➡️  No improvement this trial ({no_improvement_count[n]}/{max_trials_without_improvement})")
                    
                    # Broadcast state update
                    try:
                        asyncio.run(
                            ws_manager.broadcast_state_update(n, {
                                'n': n,
                                'score': optimized.score,
                                'side_length': optimized.side_length,
                                'iterations': optimized.iterations
                            })
                        )
                    except:
                        pass
                    
                    print()  # Blank line between puzzles
                
                # Cycle summary
                cycle_time = time.time() - cycle_start_time
                summary = self.manager.get_summary()
                active_puzzles = sum(1 for count in no_improvement_count.values() if count < max_trials_without_improvement)
                paused_puzzles = 200 - active_puzzles
                
                print(f"\n{'='*80}")
                print(f"📈 CYCLE {cycle} COMPLETE - {time.strftime('%H:%M:%S')}")
                print(f"{'='*80}")
                print(f"⏱️  Cycle Statistics:")
                print(f"   • Cycle Duration: {cycle_time:.1f} seconds ({cycle_time/60:.2f} minutes)")
                print(f"   • Puzzles Optimized: {puzzles_optimized}/200")
                print(f"   • Improvements Found: {cycle_improvements}")
                print(f"   • Average time per puzzle: {cycle_time/puzzles_optimized:.2f}s" if puzzles_optimized > 0 else "   • Average time per puzzle: N/A")
                print()
                print(f"🎯 Overall Progress:")
                print(f"   • Total Score: {summary['total_score']:.2f}")
                print(f"   • Average Score: {summary['avg_score']:.6f}")
                print(f"   • Total Iterations: {summary['total_iterations']:,}")
                print()
                print(f"📊 Puzzle Status:")
                print(f"   • Active (still optimizing): {active_puzzles}/200")
                print(f"   • Paused (50+ trials w/o improvement): {paused_puzzles}/200")
                print(f"   • Completion: {(paused_puzzles/200)*100:.1f}%")
                print(f"{'='*80}")
                
                # If all puzzles are paused, reset counters to try again
                if active_puzzles == 0:
                    print("\n" + "🔄"*30)
                    print("🔄 ALL PUZZLES PAUSED - STARTING NEW OPTIMIZATION ROUND!")
                    print("🔄 Resetting all trial counters to 0")
                    print("🔄 This allows another fresh attempt at optimization")
                    print("🔄"*30)
                    no_improvement_count = {n: 0 for n in range(1, 201)}
                
                # Broadcast progress
                try:
                    asyncio.run(ws_manager.broadcast_progress(summary))
                except:
                    pass
                
                # Auto-save every cycle (more frequent for 24/7 operation)
                if cycle % 3 == 0:
                    print(f"\n💾 AUTO-SAVE TRIGGERED (Cycle {cycle})")
                    print(f"   • Saving all puzzle states to disk...")
                    save_start = time.time()
                    self.storage.save(self.manager)
                    save_time = time.time() - save_start
                    print(f"   • Save completed in {save_time:.2f}s")
                    print(f"   • Next auto-save in 3 cycles")
                
                print(f"\n⏭️  Moving to next cycle in 2 seconds...")
                time.sleep(2)  # Brief pause between cycles
        
        # Start thread
        self.optimization_thread = threading.Thread(
            target=optimization_loop,
            daemon=True
        )
        self.optimization_thread.start()
        
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
        
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
        
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
