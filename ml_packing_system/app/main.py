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
            """Main optimization loop."""
            print("Starting optimization loop...")
            
            cycle = 0
            while self.running:
                cycle += 1
                print(f"\n{'='*50}")
                print(f"Optimization Cycle {cycle}")
                print(f"{'='*50}")
                
                # Iterate through puzzles
                for n in range(1, 201):
                    if not self.running:
                        break
                    
                    puzzle = self.manager.get_puzzle(n)
                    if puzzle is None:
                        continue
                    
                    old_score = puzzle.score
                    
                    # Optimize with callback
                    def callback(iteration, state):
                        if state.score < old_score:
                            # Broadcast improvement via WebSocket
                            try:
                                asyncio.run(
                                    ws_manager.broadcast_improvement(
                                        n, old_score, state.score
                                    )
                                )
                            except:
                                pass
                    
                    optimized = self.optimizer.optimize_puzzle(
                        puzzle,
                        max_iterations=50,
                        callback=callback
                    )
                    
                    # Update manager
                    self.manager.add_puzzle(optimized)
                    
                    # Log significant improvements
                    improvement = old_score - optimized.score
                    if improvement > 0.001:
                        print(f"  Puzzle {n:3d}: {old_score:.6f} → {optimized.score:.6f} (↓{improvement:.6f})")
                    
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
                
                # Cycle summary
                summary = self.manager.get_summary()
                print(f"\nCycle {cycle} Summary:")
                print(f"  Total Score: {summary['total_score']:.2f}")
                print(f"  Avg Score: {summary['avg_score']:.6f}")
                print(f"  Total Iterations: {summary['total_iterations']:,}")
                
                # Broadcast progress
                try:
                    asyncio.run(ws_manager.broadcast_progress(summary))
                except:
                    pass
                
                # Auto-save
                if cycle % 5 == 0:
                    print("Auto-saving...")
                    self.storage.save(self.manager)
        
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


def get_app() -> Application:
    """Get or create application instance."""
    global app_instance
    if app_instance is None:
        app_instance = Application()
        app_instance.initialize()
    return app_instance
