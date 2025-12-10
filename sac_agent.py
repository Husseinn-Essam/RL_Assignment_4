"""
Soft Actor-Critic (SAC) Agent Implementation.
Optimized for continuous action spaces with automatic entropy tuning.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from networks import ActorNetwork, CriticNetwork
from buffers import ReplayBuffer


class SACAgent:
    """Soft Actor-Critic (SAC) Agent for continuous action spaces."""
    
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.0003,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        buffer_size=1000000,
        batch_size=256,
        hidden_sizes=[256, 256],
        device='cpu',
        use_cnn=False,
        input_channels=4,
        reward_scale=1.0,
        **kwargs
    ):
        """
        Initialize SAC Agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            alpha: Entropy regularization coefficient
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            hidden_sizes: List of hidden layer sizes
            device: Device to run on ('cpu' or 'cuda')
            use_cnn: Whether to use CNN for image inputs
            input_channels: Number of input channels for CNN
            reward_scale: Scale factor for rewards
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.device = device
        self.use_cnn = use_cnn
        self.reward_scale = reward_scale
        
        # Networks
        self.actor = ActorNetwork(state_size, action_size, hidden_sizes, use_cnn, input_channels).to(device)
        
        # Two Q-networks (critics) for double Q-learning
        self.critic1 = CriticNetwork(state_size, action_size, hidden_sizes, 'q_value', use_cnn, input_channels).to(device)
        self.critic2 = CriticNetwork(state_size, action_size, hidden_sizes, 'q_value', use_cnn, input_channels).to(device)
        
        # Target networks
        self.target_critic1 = CriticNetwork(state_size, action_size, hidden_sizes, 'q_value', use_cnn, input_channels).to(device)
        self.target_critic2 = CriticNetwork(state_size, action_size, hidden_sizes, 'q_value', use_cnn, input_channels).to(device)
        
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        self.steps_done = 0
    
    def select_action(self, state, epsilon=None):
        """
        Select action using the current policy.
        
        Args:
            state: Current state
            epsilon: Not used for SAC (for compatibility)
        
        Returns:
            Selected action
        """
        if self.use_cnn:
            # For CNN, state should be [C, H, W], add batch dimension to make [1, C, H, W]
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) / 255.0
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mean, log_std = self.actor(state_tensor)
            std = log_std.exp()
            dist = Normal(mean, std)
            u = dist.rsample()
            action = torch.tanh(u)  # Bound action to [-1,1] scaled policy space
            return action.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.memory.push(state, action, reward * self.reward_scale, next_state, done)
        self.steps_done += 1
    
    def train(self):
        """Train the agent on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Normalize image inputs
        if self.use_cnn:
            states = states.float() / 255.0
            next_states = next_states.float() / 255.0
        
        # Update critics
        with torch.no_grad():
            # Reparameterization trick with tanh correction 
            next_mean, next_log_std = self.actor(next_states)
            next_std = next_log_std.exp()
            next_dist = Normal(next_mean, next_std)
            u_next = next_dist.rsample()
            a_next = torch.tanh(u_next)
            # Tanh change-of-variables correction for log-prob
            log_pi_next = (
                next_dist.log_prob(u_next).sum(dim=-1)
                - torch.log(1 - torch.tanh(u_next).pow(2) + 1e-6).sum(dim=-1)
            )
            
            next_q1 = self.target_critic1(next_states, a_next).squeeze()
            next_q2 = self.target_critic2(next_states, a_next).squeeze()
            next_q = torch.min(next_q1, next_q2) - self.alpha * log_pi_next
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        current_q1 = self.critic1(states, actions).squeeze()
        current_q2 = self.critic2(states, actions).squeeze()
        
        critic1_loss = F.smooth_l1_loss(current_q1, target_q)
        critic2_loss = F.smooth_l1_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        mean, log_std = self.actor(states)
        std = log_std.exp()
        dist = Normal(mean, std)
        u = dist.rsample()
        a = torch.tanh(u)
        log_pi = (
            dist.log_prob(u).sum(dim=-1)
            - torch.log(1 - torch.tanh(u).pow(2) + 1e-6).sum(dim=-1)
        )
        
        q1 = self.critic1(states, a).squeeze()
        q2 = self.critic2(states, a).squeeze()
        min_q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_pi - min_q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return (critic1_loss.item() + critic2_loss.item() + actor_loss.item()) / 3
    
    def get_entropy_coef(self):
        """Return alpha (entropy coefficient) for SAC."""
        return self.alpha
    
    def save(self, filepath):
        """Save model weights."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'steps_done': self.steps_done
        }, filepath)
    
    def load(self, filepath):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        self.steps_done = checkpoint.get('steps_done', 0)
