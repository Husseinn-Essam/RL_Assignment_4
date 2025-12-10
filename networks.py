"""
Neural Network Architectures for RL Agents.
Contains Actor and Critic networks optimized for continuous action spaces.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorNetwork(nn.Module):
    """Actor Network for continuous action spaces."""
    
    def __init__(self, state_size, action_size, hidden_sizes=[256, 256], use_cnn=False, input_channels=4):
        """
        Initialize the actor network.
        
        Args:
            state_size: Dimension of state space (or image size if use_cnn=True)
            action_size: Dimension of action space
            hidden_sizes: List of hidden layer sizes
            use_cnn: Whether to use CNN for image inputs
            input_channels: Number of input channels for CNN
        """
        super(ActorNetwork, self).__init__()
        
        self.use_cnn = use_cnn
        
        if use_cnn:
            # CNN feature extractor for image inputs
            self.cnn = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
            # Calculate CNN output size
            with torch.no_grad():
                sample_input = torch.zeros(1, input_channels, 84, 84)
                cnn_output_size = self.cnn(sample_input).shape[1]
            input_size = cnn_output_size
        else:
            input_size = state_size
        
        # Build MLP layers
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        self.base = nn.Sequential(*layers)
        
        # Output mean for continuous actions
        self.mean_head = nn.Linear(input_size, action_size)
        # Learnable log_std parameter (state-independent for stability)
        self.log_std = nn.Parameter(torch.zeros(action_size))
    
    def forward(self, x):
        """Forward pass through the network."""
        if self.use_cnn:
            x = self.cnn(x)
        x = self.base(x)
        
        # Bound mean to action range using tanh
        mean = torch.tanh(self.mean_head(x))
        # Use learnable but state-independent log_std for stability
        log_std = self.log_std.expand_as(mean)
        log_std = torch.clamp(log_std, -2, 0.5)  # Tighter range for stability
        return mean, log_std


class CriticNetwork(nn.Module):
    """Critic Network for value function estimation."""
    
    def __init__(self, state_size, action_size=None, hidden_sizes=[256, 256], output_type='value', use_cnn=False, input_channels=4):
        """
        Initialize the critic network.
        
        Args:
            state_size: Dimension of state space (or image size if use_cnn=True)
            action_size: Dimension of action space (for Q-value estimation)
            hidden_sizes: List of hidden layer sizes
            output_type: 'value' for V(s) or 'q_value' for Q(s,a)
            use_cnn: Whether to use CNN for image inputs
            input_channels: Number of input channels for CNN
        """
        super(CriticNetwork, self).__init__()
        
        self.output_type = output_type
        self.use_cnn = use_cnn
        
        if use_cnn:
            # CNN feature extractor for image inputs
            self.cnn = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
            # Calculate CNN output size
            with torch.no_grad():
                sample_input = torch.zeros(1, input_channels, 84, 84)
                cnn_output_size = self.cnn(sample_input).shape[1]
            
            if output_type == 'q_value' and action_size is not None:
                input_size = cnn_output_size + action_size
            else:
                input_size = cnn_output_size
        else:
            if output_type == 'q_value' and action_size is not None:
                input_size = state_size + action_size
            else:
                input_size = state_size
        
        # Build MLP layers
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        # Output layer - single value
        layers.append(nn.Linear(input_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state, action=None):
        """Forward pass through the network."""
        if self.use_cnn:
            state_features = self.cnn(state)
            if self.output_type == 'q_value' and action is not None:
                x = torch.cat([state_features, action], dim=-1)
            else:
                x = state_features
        else:
            if self.output_type == 'q_value' and action is not None:
                x = torch.cat([state, action], dim=-1)
            else:
                x = state
        return self.network(x)
