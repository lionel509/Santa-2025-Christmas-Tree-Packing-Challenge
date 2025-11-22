"""PPO (Proximal Policy Optimization) Agent."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple
from .network import PolicyNetwork, ValueNetwork
from .environment import PackingEnv


class PPOAgent:
    """PPO agent for tree packing optimization."""
    
    def __init__(
        self,
        state_dim: int,
        num_trees: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = 'cpu'
    ):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: State space dimension
            num_trees: Number of trees in puzzle
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            device: Device for computation
        """
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Networks
        self.policy = PolicyNetwork(state_dim, num_trees=num_trees).to(device)
        self.value = ValueNetwork(state_dim).to(device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        
        # Training stats
        self.training_steps = 0
        self.episode_count = 0
    
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """
        Select action from current policy.
        
        Args:
            state: State tensor
            deterministic: If True, use mean action
        
        Returns:
            (action, log_prob) tuple
        """
        with torch.no_grad():
            action, log_prob = self.policy.get_action(state, deterministic)
        
        return action.cpu().numpy().flatten(), log_prob.item()
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
        
        Returns:
            (advantages, returns) tuple
        """
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + np.array(values, dtype=np.float32)
        
        return advantages, returns
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        epochs: int = 4,
        batch_size: int = 64
    ) -> dict:
        """
        Update policy and value networks using PPO.
        
        Args:
            states: State tensor
            actions: Action tensor
            old_log_probs: Old log probabilities
            advantages: Advantage estimates
            returns: Return estimates
            epochs: Number of update epochs
            batch_size: Mini-batch size
        
        Returns:
            Dictionary of training metrics
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = states.shape[0]
        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'total_loss': 0.0
        }
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(dataset_size)
            
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get current policy distribution
                mean, log_std = self.policy(batch_states)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mean, std)
                
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                values = self.value(batch_states)
                value_loss = nn.MSELoss()(values, batch_returns)
                
                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update policy
                self.policy_optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.policy_optimizer.step()
                
                # Update value
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
                self.value_optimizer.step()
                
                # Track metrics
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropy.item()
                metrics['total_loss'] += total_loss.item()
        
        # Average metrics
        num_updates = epochs * (dataset_size // batch_size + 1)
        for key in metrics:
            metrics[key] /= num_updates
        
        self.training_steps += 1
        return metrics
    
    def save(self, path: str):
        """Save agent checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'training_steps': self.training_steps,
            'episode_count': self.episode_count
        }, path)
    
    def load(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.training_steps = checkpoint.get('training_steps', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
