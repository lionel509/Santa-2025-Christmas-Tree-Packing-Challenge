"""Hybrid optimizer combining ML and heuristics."""

import random
import time
import torch
import numpy as np
from typing import Optional, Callable
from ..state import PuzzleState, PuzzleManager
from ..ml import PPOAgent, PackingEnv
from .heuristics import local_search, simulated_annealing, squeeze_bounds, jitter_positions
from .initializer import initialize_puzzle


class HybridOptimizer:
    """Combines RL agent with heuristic optimizers."""
    
    def __init__(
        self,
        agent: Optional[PPOAgent] = None,
        device: str = 'cpu',
        use_ml: bool = True,
        use_heuristics: bool = True
    ):
        """
        Initialize hybrid optimizer.
        
        Args:
            agent: PPO agent (optional)
            device: Device for computation
            use_ml: Whether to use ML optimization
            use_heuristics: Whether to use heuristic optimization
        """
        self.agent = agent
        self.device = device
        self.use_ml = use_ml
        self.use_heuristics = use_heuristics
        self.iteration_count = 0
        
    def optimize_puzzle(
        self,
        state: PuzzleState,
        max_iterations: int = 100,
        callback: Optional[Callable] = None,
        verbose: bool = False
    ) -> PuzzleState:
        """
        Optimize a single puzzle.
        
        Args:
            state: Initial puzzle state
            max_iterations: Maximum optimization iterations
            callback: Optional callback function(iteration, state)
            verbose: Enable detailed progress logging
        
        Returns:
            Optimized puzzle state
        """
        current = state.copy()
        best = current.copy()
        best_score = current.score
        improvements_found = 0
        ml_attempts = 0
        heuristic_attempts = 0
        
        for iteration in range(max_iterations):
            # Choose optimization method
            method = random.choice(['ml', 'heuristic']) if self.use_ml and self.use_heuristics else (
                'ml' if self.use_ml else 'heuristic'
            )
            
            if verbose and iteration % 20 == 0:
                print(f"      Iteration {iteration}/{max_iterations}: score={best_score:.6f}, improvements={improvements_found}")
            
            if method == 'ml' and self.agent is not None:
                # RL-based optimization
                ml_attempts += 1
                current = self._optimize_with_ml(current, steps=10)
            else:
                # Heuristic-based optimization
                heuristic_attempts += 1
                current = self._optimize_with_heuristics(current)
            
            # Update best
            if current.score < best_score:
                old_best = best_score
                best = current.copy()
                best_score = current.score
                best.last_improvement = time.time()
                improvements_found += 1
                
                if verbose:
                    improvement = old_best - best_score
                    print(f"      ⭐ Iteration {iteration}: IMPROVEMENT {old_best:.6f} → {best_score:.6f} (↓{improvement:.6f})")
            
            best.iterations += 1
            self.iteration_count += 1
            
            # Callback
            if callback is not None:
                callback(iteration, best)
            
            # Occasionally reset to best
            if iteration % 20 == 19:
                current = best.copy()
        
        if verbose:
            print(f"      Completed: ML attempts={ml_attempts}, Heuristic attempts={heuristic_attempts}, Total improvements={improvements_found}")
        
        return best
    
    def _optimize_with_ml(self, state: PuzzleState, steps: int = 10) -> PuzzleState:
        """Optimize using RL agent."""
        if self.agent is None:
            return state
        
        env = PackingEnv(state, max_steps=steps)
        obs = env.reset()
        
        for _ in range(steps):
            # Get action from agent
            state_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            action, _ = self.agent.select_action(state_tensor, deterministic=False)
            
            # Take step
            obs, reward, done, info = env.step(action)
            
            if done:
                break
        
        return env.get_state()
    
    def _optimize_with_heuristics(self, state: PuzzleState) -> PuzzleState:
        """Optimize using heuristics."""
        # Choose random heuristic
        method = random.choice(['local_search', 'simulated_annealing', 'squeeze', 'jitter'])
        
        if method == 'local_search':
            return local_search(state, iterations=20, step_size=0.03)
        elif method == 'simulated_annealing':
            return simulated_annealing(state, iterations=50, initial_temp=0.5)
        elif method == 'squeeze':
            return squeeze_bounds(state, factor=0.99)
        elif method == 'jitter':
            return jitter_positions(state, magnitude=0.01)
        else:
            return state
    
    def optimize_all_puzzles(
        self,
        manager: PuzzleManager,
        iterations_per_puzzle: int = 100,
        callback: Optional[Callable] = None
    ):
        """
        Continuously optimize all puzzles.
        
        Args:
            manager: Puzzle manager
            iterations_per_puzzle: Iterations per puzzle per cycle
            callback: Optional callback function(n, state)
        """
        cycle = 0
        
        while True:
            cycle += 1
            print(f"\n=== Optimization Cycle {cycle} ===")
            
            # Iterate through all puzzles
            for n in range(1, 201):
                puzzle = manager.get_puzzle(n)
                
                if puzzle is None:
                    # Initialize if missing
                    puzzle = initialize_puzzle(n, method='greedy')
                    manager.add_puzzle(puzzle)
                
                # Optimize
                old_score = puzzle.score
                optimized = self.optimize_puzzle(
                    puzzle,
                    max_iterations=iterations_per_puzzle
                )
                
                # Update manager
                manager.add_puzzle(optimized)
                
                # Log improvement
                improvement = old_score - optimized.score
                if improvement > 0:
                    print(f"  Puzzle {n}: {old_score:.6f} -> {optimized.score:.6f} (↓ {improvement:.6f})")
                
                # Callback
                if callback is not None:
                    callback(n, optimized)
            
            # Summary
            summary = manager.get_summary()
            print(f"\nCycle {cycle} complete:")
            print(f"  Total score: {summary['total_score']:.2f}")
            print(f"  Average score: {summary['avg_score']:.6f}")
            print(f"  Total iterations: {summary['total_iterations']}")


class AdaptiveOptimizer(HybridOptimizer):
    """Adaptive optimizer that adjusts strategy based on progress."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.success_rates = {
            'ml': 0.5,
            'local_search': 0.5,
            'simulated_annealing': 0.5,
            'squeeze': 0.5,
            'jitter': 0.5
        }
        self.method_counts = {key: 0 for key in self.success_rates}
        self.method_successes = {key: 0 for key in self.success_rates}
    
    def _optimize_with_heuristics(self, state: PuzzleState) -> PuzzleState:
        """Choose heuristic based on success rates."""
        # Weighted random choice
        methods = list(self.success_rates.keys())
        weights = [self.success_rates[m] + 0.1 for m in methods]  # Add epsilon for exploration
        
        method = random.choices(methods, weights=weights)[0]
        
        self.method_counts[method] += 1
        old_score = state.score
        
        # Apply method
        if method == 'ml':
            result = self._optimize_with_ml(state, steps=10)
        elif method == 'local_search':
            result = local_search(state, iterations=20)
        elif method == 'simulated_annealing':
            result = simulated_annealing(state, iterations=50)
        elif method == 'squeeze':
            result = squeeze_bounds(state)
        elif method == 'jitter':
            result = jitter_positions(state)
        else:
            result = state
        
        # Update success rate
        if result.score < old_score:
            self.method_successes[method] += 1
        
        if self.method_counts[method] > 0:
            self.success_rates[method] = self.method_successes[method] / self.method_counts[method]
        
        return result
