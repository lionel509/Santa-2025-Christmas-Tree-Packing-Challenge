"""Neural network architectures for RL agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PolicyNetwork(nn.Module):
    """Actor network that outputs action distributions."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256, num_trees: int = 1):
        """
        Initialize policy network.
        
        Args:
            state_dim: Dimension of state space
            hidden_dim: Hidden layer size
            num_trees: Number of trees (for action space sizing)
        """
        super().__init__()
        
        self.num_trees = num_trees
        
        # Shared feature extraction
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Action heads (dx, dy, ddeg) for each tree
        # Output: mean and log_std for continuous actions
        self.action_mean = nn.Linear(hidden_dim // 2, num_trees * 3)
        self.action_log_std = nn.Linear(hidden_dim // 2, num_trees * 3)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: State tensor [batch_size, state_dim]
        
        Returns:
            (action_mean, action_log_std) tuple
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mean = torch.tanh(self.action_mean(x)) * 0.1  # Small movements
        log_std = self.action_log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # Prevent extreme values
        
        return mean, log_std
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: State tensor
            deterministic: If True, return mean action
        
        Returns:
            (action, log_prob) tuple
        """
        mean, log_std = self.forward(state)
        
        if deterministic:
            return mean, torch.zeros_like(mean)
        
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        action = normal.sample()
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob


class ValueNetwork(nn.Module):
    """Critic network that estimates state values."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        """
        Initialize value network.
        
        Args:
            state_dim: Dimension of state space
            hidden_dim: Hidden layer size
        """
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.value_head = nn.Linear(hidden_dim // 2, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor [batch_size, state_dim]
        
        Returns:
            Value estimate [batch_size, 1]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.value_head(x)
        
        return value


class CombinedNetwork(nn.Module):
    """Combined actor-critic network for more efficient training."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256, num_trees: int = 1):
        super().__init__()
        
        self.num_trees = num_trees
        
        # Shared layers
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor head
        self.actor_fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.action_mean = nn.Linear(hidden_dim // 2, num_trees * 3)
        self.action_log_std = nn.Linear(hidden_dim // 2, num_trees * 3)
        
        # Critic head
        self.critic_fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.value_head = nn.Linear(hidden_dim // 2, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for both actor and critic.
        
        Returns:
            (action_mean, action_log_std, value) tuple
        """
        # Shared features
        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))
        
        # Actor
        actor_x = F.relu(self.actor_fc(x))
        mean = torch.tanh(self.action_mean(actor_x)) * 0.1
        log_std = torch.clamp(self.action_log_std(actor_x), -20, 2)
        
        # Critic
        critic_x = F.relu(self.critic_fc(x))
        value = self.value_head(critic_x)
        
        return mean, log_std, value
