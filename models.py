"""
SAC, PPO, and TD3 Implementation
This module implements Soft Actor-Critic (SAC), Proximal Policy Optimization (PPO),
and Twin Delayed DDPG (TD3) algorithms for reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np
import random
import math
from collections import deque, namedtuple


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """Experience Replay Buffer for storing and sampling transitions."""
    
    def __init__(self, capacity):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add an experience to the buffer."""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor(np.array([e.state for e in experiences]))
        actions = torch.FloatTensor(np.array([e.action for e in experiences]))
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences]))
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences]))
        dones = torch.FloatTensor(np.array([e.done for e in experiences]))
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)


class ActorNetwork(nn.Module):
    """Actor Network for policy-based methods."""
    
    def __init__(self, state_size, action_size, hidden_sizes=[256, 256], discrete=True, use_cnn=False, input_channels=4):
        """
        Initialize the actor network.
        
        Args:
            state_size: Dimension of state space (or image size if use_cnn=True)
            action_size: Dimension of action space
            hidden_sizes: List of hidden layer sizes
            discrete: Whether action space is discrete
            use_cnn: Whether to use CNN for image inputs
            input_channels: Number of input channels for CNN
        """
        super(ActorNetwork, self).__init__()
        
        self.discrete = discrete
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
        
        if discrete:
            # Output action probabilities for discrete actions
            self.action_head = nn.Linear(input_size, action_size)
        else:
            # Output mean for continuous actions
            self.mean_head = nn.Linear(input_size, action_size)
            # Learnable log_std parameter (state-independent for stability)
            self.log_std = nn.Parameter(torch.zeros(action_size))
    
    def forward(self, x):
        """Forward pass through the network."""
        if self.use_cnn:
            x = self.cnn(x)
        x = self.base(x)
        
        if self.discrete:
            # Return raw logits - let Categorical handle softmax for numerical stability
            logits = self.action_head(x)
            return logits
        else:
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
        discrete=True,
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
            discrete: Whether action space is discrete
            use_cnn: Whether to use CNN for image inputs
            input_channels: Number of input channels for CNN
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.device = device
        self.discrete = discrete
        self.use_cnn = use_cnn
        self.reward_scale = reward_scale
        
        # Networks
        self.actor = ActorNetwork(state_size, action_size, hidden_sizes, discrete, use_cnn, input_channels).to(device)
        
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
            if self.discrete:
                logits = self.actor(state_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample()
                return action.item()
            else:
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
        
        # For discrete actions, convert to one-hot
        if self.discrete:
            actions = actions.long()
            actions_one_hot = F.one_hot(actions, self.action_size).float()
        else:
            actions_one_hot = actions
        
        # Update critics
        with torch.no_grad():
            if self.discrete:
                # For discrete SAC, use expected value over all actions
                next_action_probs = self.actor(next_states)
                next_log_probs = torch.log(next_action_probs + 1e-8)
                
                # Compute Q-values for all actions
                next_q1_all = torch.zeros(next_states.size(0), self.action_size).to(self.device)
                next_q2_all = torch.zeros(next_states.size(0), self.action_size).to(self.device)
                
                batch_n = next_states.size(0)
                for a in range(self.action_size):
                    a_idx = torch.full((batch_n,), a, device=self.device, dtype=torch.long)
                    action_one_hot = F.one_hot(a_idx, self.action_size).float()
                    next_q1_all[:, a] = self.target_critic1(next_states, action_one_hot).squeeze()
                    next_q2_all[:, a] = self.target_critic2(next_states, action_one_hot).squeeze()
                
                # Expected value: sum over actions weighted by policy
                next_q = torch.min(next_q1_all, next_q2_all)
                next_v = (next_action_probs * (next_q - self.alpha * next_log_probs)).sum(dim=1)
                target_q = rewards + (1 - dones) * self.gamma * next_v
            else:
                # Continuous SAC: reparameterization trick with tanh correction 
                next_mean, next_log_std = self.actor(next_states)
                next_std = next_log_std.exp()
                next_dist = Normal(next_mean, next_std)
                u_next = next_dist.rsample()
                a_next = torch.tanh(u_next)
                # Tanh change-of-variables correction for log-prob
                log_pi_next = (
                    next_dist.log_prob(u_next).sum(dim=-1)
                    - torch.log(1 - torch.tanh(u_next).pow(2)).sum(dim=-1)
                )
                
                next_q1 = self.target_critic1(next_states, a_next).squeeze()
                next_q2 = self.target_critic2(next_states, a_next).squeeze()
                next_q = torch.min(next_q1, next_q2) - self.alpha * log_pi_next
                target_q = rewards + (1 - dones) * self.gamma * next_q
        
        current_q1 = self.critic1(states, actions_one_hot).squeeze()
        current_q2 = self.critic2(states, actions_one_hot).squeeze()
        
        critic1_loss = F.smooth_l1_loss(current_q1, target_q)
        critic2_loss = F.smooth_l1_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        if self.discrete:
            # For discrete actions, compute expected Q-value over all actions
            logits = self.actor(states)
            action_probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)  # More stable than log(softmax)
            
            # Compute Q-values for all actions
            q1_all = torch.zeros(states.size(0), self.action_size).to(self.device)
            q2_all = torch.zeros(states.size(0), self.action_size).to(self.device)
            
            batch_n = states.size(0)
            for a in range(self.action_size):
                a_idx = torch.full((batch_n,), a, device=self.device, dtype=torch.long)
                action_one_hot = F.one_hot(a_idx, self.action_size).float()
                q1_all[:, a] = self.critic1(states, action_one_hot).squeeze()
                q2_all[:, a] = self.critic2(states, action_one_hot).squeeze()
            
            min_q_all = torch.min(q1_all, q2_all)
            
            # Policy loss: maximize expected value
            # Maximize expected Q minus entropy term => minimize (alpha*log_pi - Q)
            actor_loss = (action_probs * (self.alpha * log_probs - min_q_all)).sum(dim=1).mean()
        else:
            # Continuous SAC actor update with reparameterization and tanh correction
            mean, log_std = self.actor(states)
            std = log_std.exp()
            dist = Normal(mean, std)
            u = dist.rsample()
            a = torch.tanh(u)
            epsilon = 1e-6
            log_pi = (
                dist.log_prob(u).sum(dim=-1)
                - torch.log(1 - torch.tanh(u).pow(2) + epsilon).sum(dim=-1)
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
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        self.steps_done = checkpoint.get('steps_done', 0)


class PPOAgent:
    """Proximal Policy Optimization (PPO) Agent with Clipped Objective and GAE."""
    
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        epsilon_clip=0.2,
        epochs=10,
        batch_size=64,
        hidden_sizes=[256, 256],
        device='cpu',
        discrete=True,
        entropy_coef=0.05,
        entropy_decay_episodes=500,
        max_grad_norm=0.5,
        action_scale=2.0,
        reward_scale=1.0,
        use_cnn=False,
        input_channels=4,
        **kwargs
    ):
        """
        Initialize PPO Agent with Generalized Advantage Estimation (GAE).
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for returns
            gae_lambda: GAE lambda parameter (0=TD, 1=Monte-Carlo)
                       - lambda=0: High bias, low variance (like 1-step TD)
                       - lambda=1: Low bias, high variance (like Monte-Carlo)
                       - lambda=0.95-0.99: Good balance (recommended)
            epsilon_clip: Clipping parameter for PPO objective
            epochs: Number of optimization epochs per update
            batch_size: Batch size for training
            hidden_sizes: List of hidden layer sizes
            device: Device to run on ('cpu' or 'cuda')
            discrete: Whether action space is discrete
            entropy_coef: Initial entropy coefficient for exploration bonus
            entropy_decay_episodes: Number of episodes to linearly decay entropy to zero
            max_grad_norm: Max gradient norm for clipping
            action_scale: Scale for continuous actions (e.g., 2.0 for Pendulum)
            reward_scale: Scale factor for rewards (helps stabilize training)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon_clip = epsilon_clip
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.discrete = discrete
        self.initial_entropy_coef = entropy_coef
        self.entropy_coef = entropy_coef
        self.entropy_decay_episodes = entropy_decay_episodes
        self.entropy_decay_rate = entropy_coef / max(1, entropy_decay_episodes)  # Linear decay per episode
        self.max_grad_norm = max_grad_norm
        self.action_scale = action_scale
        self.reward_scale = reward_scale if not discrete else 1.0  # Only scale rewards for continuous
        self.episode_count = 0
        self.use_cnn = use_cnn
        
        # Actor and Critic networks
        self.actor = ActorNetwork(state_size, action_size, hidden_sizes, discrete, use_cnn, input_channels).to(device)
        self.critic = CriticNetwork(state_size, hidden_sizes=hidden_sizes, output_type='value', use_cnn=use_cnn, input_channels=input_channels).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Storage for trajectory
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []  # Needed for GAE bootstrapping
        self.dones = []
        
        self.steps_done = 0
    
    def select_action(self, state, epsilon=None):
        """
        Select action using the current policy.
        
        Args:
            state: Current state
            epsilon: Not used for PPO (for compatibility)
        
        Returns:
            Selected action
        """
        if self.use_cnn:
            # For CNN, state should be [C, H, W], add batch dimension to make [1, C, H, W]
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) / 255.0
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.discrete:
                logits = self.actor(state_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample()
                return action.item()
            else:
                mean, log_std = self.actor(state_tensor)
                std = log_std.exp()
                dist = Normal(mean, std)
                action = dist.sample()
                # Clip to valid range and scale
                action = torch.clamp(action, -1.0, 1.0) * self.action_scale
                return action.cpu().numpy().flatten()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition for training."""
        self.states.append(state)
        # Store action normalized to [-1, 1] for log_prob computation
        if not self.discrete:
            self.actions.append(action / self.action_scale)
        else:
            self.actions.append(action)
        # Scale rewards for continuous control (helps with large negative rewards)
        self.rewards.append(reward * self.reward_scale)
        self.next_states.append(next_state)  # Store for GAE
        self.dones.append(done)
        
        self.steps_done += 1
    
    def train(self):
        """Train the agent using PPO clipped objective with GAE."""
        if len(self.states) == 0:
            return None
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)
        
        # Normalize image inputs
        if self.use_cnn:
            states = states.float() / 255.0
            next_states = next_states.float() / 255.0
        
        if self.discrete:
            actions = torch.LongTensor(self.actions).to(self.device)
        else:
            # Handle both 1D and multi-D continuous actions
            actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
            if actions.dim() == 1:
                actions = actions.unsqueeze(-1)  # [batch] -> [batch, 1]
        
        # Get old log probs and values for PPO
        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            # Handle single step case
            if values.dim() == 0:
                values = values.unsqueeze(0)
            if next_values.dim() == 0:
                next_values = next_values.unsqueeze(0)
            
            if self.discrete:
                logits = self.actor(states)
                dist = Categorical(logits=logits)
                old_log_probs = dist.log_prob(actions)
            else:
                mean, log_std = self.actor(states)
                std = log_std.exp()
                dist = Normal(mean, std)
                # actions are normalized to [-1, 1], directly compute log_prob
                old_log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # Compute GAE (Generalized Advantage Estimation)
        # A_t = delta_t + (gamma * lambda) * delta_{t+1} + ... + (gamma * lambda)^{T-t-1} * delta_{T-1}
        # where delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        returns = torch.zeros(T, device=self.device)
        
        # Compute TD errors (deltas)
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        
        # Compute GAE advantages using reverse cumulative sum
        gae = 0.0
        for t in reversed(range(T)):
            # GAE: A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
            gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        # Compute returns: R_t = A_t + V(s_t)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for multiple epochs
        total_actor_loss = 0
        total_critic_loss = 0
        
        for _ in range(self.epochs):
            # Get current log probabilities and values
            if self.discrete:
                logits = self.actor(states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
            else:
                mean, log_std = self.actor(states)
                std = log_std.exp()
                dist = Normal(mean, std)
                # actions are normalized to [-1, 1], directly compute log_prob
                new_log_probs = dist.log_prob(actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
            
            new_values = self.critic(states).squeeze()
            if new_values.dim() == 0:
                new_values = new_values.unsqueeze(0)
            
            # Compute ratio (pi_theta / pi_theta_old)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Compute surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            
            # Actor loss (PPO clipped objective) with entropy bonus
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            
            # Critic loss - SmoothL1Loss is more robust than MSE
            critic_loss = F.smooth_l1_loss(new_values, returns.detach())
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
        
        # Linear decay of entropy coefficient to zero
        self.episode_count += 1
        if self.episode_count < self.entropy_decay_episodes:
            self.entropy_coef = self.initial_entropy_coef - (self.entropy_decay_rate * self.episode_count)
        else:
            self.entropy_coef = 0.0
        
        avg_loss = (total_actor_loss + total_critic_loss) / (2 * self.epochs)
        
        # Clear trajectory
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
        return avg_loss
    
    def get_entropy_coef(self):
        """Return current entropy coefficient."""
        return self.entropy_coef
    
    def save(self, filepath):
        """Save model weights."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'steps_done': self.steps_done
        }, filepath)
    
    def load(self, filepath):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.steps_done = checkpoint.get('steps_done', 0)


class TD3Agent:
    """
    Twin Delayed DDPG (TD3) Agent for continuous action spaces.
    
    TD3 improves upon DDPG with three key techniques:
    1. Twin Q-networks (Clipped Double Q-learning) to reduce overestimation bias
    2. Delayed policy updates - update policy less frequently than Q-functions
    3. Target policy smoothing - add noise to target actions for robustness
    """
    
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.0003,
        gamma=0.99,
        tau=0.005,
        buffer_size=1000000,
        batch_size=256,
        hidden_sizes=[256, 256],
        device='cpu',
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        exploration_noise=0.1,
        use_cnn=False,
        input_channels=4,
        reward_scale=1.0,
        **kwargs
    ):
        """
        Initialize TD3 Agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            hidden_sizes: List of hidden layer sizes
            device: Device to run on ('cpu' or 'cuda')
            policy_noise: Std of Gaussian noise added to target policy (smoothing)
            noise_clip: Range to clip target policy noise
            policy_delay: Frequency of delayed policy updates
            exploration_noise: Std of exploration noise added to actions
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.exploration_noise = exploration_noise
        self.use_cnn = use_cnn
        self.reward_scale = reward_scale
        
        # Actor network (deterministic policy)
        self.actor = ActorNetwork(state_size, action_size, hidden_sizes, discrete=False, use_cnn=use_cnn, input_channels=input_channels).to(device)
        self.actor_target = ActorNetwork(state_size, action_size, hidden_sizes, discrete=False, use_cnn=use_cnn, input_channels=input_channels).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Twin Q-networks (critics)
        self.critic1 = CriticNetwork(state_size, action_size, hidden_sizes, 'q_value', use_cnn=use_cnn, input_channels=input_channels).to(device)
        self.critic2 = CriticNetwork(state_size, action_size, hidden_sizes, 'q_value', use_cnn=use_cnn, input_channels=input_channels).to(device)
        
        # Target networks
        self.target_critic1 = CriticNetwork(state_size, action_size, hidden_sizes, 'q_value', use_cnn=use_cnn, input_channels=input_channels).to(device)
        self.target_critic2 = CriticNetwork(state_size, action_size, hidden_sizes, 'q_value', use_cnn=use_cnn, input_channels=input_channels).to(device)
        
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        self.steps_done = 0
        self.total_it = 0  # Total training iterations for delayed policy updates
    
    def select_action(self, state, epsilon=None):
        """
        Select action using the deterministic policy with exploration noise.
        
        Args:
            state: Current state
            epsilon: Not used for TD3 (for compatibility)
        
        Returns:
            Selected action with exploration noise
        """
        if self.use_cnn:
            # For CNN, state should be [C, H, W], add batch dimension to make [1, C, H, W]
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) / 255.0
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # TD3 uses deterministic policy (only mean, no log_std)
            mean, _ = self.actor(state_tensor)
            action = mean.cpu().numpy()[0]
            
            # Add exploration noise during training
            if epsilon is None or epsilon > 0:  # Training mode
                noise = np.random.normal(0, self.exploration_noise, size=self.action_size)
                action = action + noise
                action = np.clip(action, -1.0, 1.0)
            
            return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.memory.push(state, action, reward * self.reward_scale, next_state, done)
        self.steps_done += 1
    
    def train(self):
        """Train the agent on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return None
        
        self.total_it += 1
        
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
        
        # Ensure actions are 2D [batch, action_dim]
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        
        # Update critics
        with torch.no_grad():
            # Target policy smoothing: add clipped noise to target actions
            next_mean, _ = self.actor_target(next_states)
            noise = torch.randn_like(next_mean) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_action = next_mean + noise
            next_action = torch.clamp(next_action, -1.0, 1.0)
            
            # Compute target Q-values using clipped double Q-learning
            target_q1 = self.target_critic1(next_states, next_action).squeeze()
            target_q2 = self.target_critic2(next_states, next_action).squeeze()
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Current Q estimates
        current_q1 = self.critic1(states, actions).squeeze()
        current_q2 = self.critic2(states, actions).squeeze()
        
        # Critic loss
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # Update critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()
        
        # Update critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()
        
        # Delayed policy updates
        actor_loss = torch.tensor(0.0)
        if self.total_it % self.policy_delay == 0:
            # Compute actor loss (maximize Q-value)
            mean, _ = self.actor(states)
            actor_loss = -self.critic1(states, mean).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return (critic1_loss.item() + critic2_loss.item() + actor_loss.item()) / 3
    
    def get_entropy_coef(self):
        """Return exploration noise for TD3 (for compatibility with logging)."""
        return self.exploration_noise
    
    def save(self, filepath):
        """Save model weights."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'steps_done': self.steps_done,
            'total_it': self.total_it
        }, filepath)
    
    def load(self, filepath):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        self.steps_done = checkpoint.get('steps_done', 0)
        self.total_it = checkpoint.get('total_it', 0)

