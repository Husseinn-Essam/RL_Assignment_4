"""
Proximal Policy Optimization (PPO) Agent Implementation.
Uses clipped surrogate objective and Generalized Advantage Estimation (GAE).
Optimized for continuous action spaces.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from networks import ActorNetwork, CriticNetwork


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
        entropy_coef=0.05,
        entropy_decay_episodes=500,
        max_grad_norm=0.5,
        action_scale=1.0,
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
            epsilon_clip: Clipping parameter for PPO objective
            epochs: Number of optimization epochs per update
            batch_size: Batch size for training
            hidden_sizes: List of hidden layer sizes
            device: Device to run on ('cpu' or 'cuda')
            entropy_coef: Initial entropy coefficient for exploration bonus
            entropy_decay_episodes: Number of episodes to linearly decay entropy to zero
            max_grad_norm: Max gradient norm for clipping
            action_scale: Scale for continuous actions
            reward_scale: Scale factor for rewards (helps stabilize training)
            use_cnn: Whether to use CNN for image inputs
            input_channels: Number of input channels for CNN
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon_clip = epsilon_clip
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.initial_entropy_coef = entropy_coef
        self.entropy_coef = entropy_coef
        self.entropy_decay_episodes = entropy_decay_episodes
        self.entropy_decay_rate = entropy_coef / max(1, entropy_decay_episodes)
        self.max_grad_norm = max_grad_norm
        self.action_scale = action_scale
        self.reward_scale = reward_scale
        self.episode_count = 0
        self.use_cnn = use_cnn
        
        # Actor and Critic networks
        self.actor = ActorNetwork(state_size, action_size, hidden_sizes, use_cnn, input_channels).to(device)
        self.critic = CriticNetwork(state_size, hidden_sizes=hidden_sizes, output_type='value', use_cnn=use_cnn, input_channels=input_channels).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Storage for trajectory
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
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
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) / 255.0
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
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
        self.actions.append(action / self.action_scale)
        # Scale rewards for continuous control
        self.rewards.append(reward * self.reward_scale)
        self.next_states.append(next_state)
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
        
        # Handle both 1D and multi-D continuous actions
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        
        # Get old log probs and values for PPO
        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            # Handle single step case
            if values.dim() == 0:
                values = values.unsqueeze(0)
            if next_values.dim() == 0:
                next_values = next_values.unsqueeze(0)
            
            mean, log_std = self.actor(states)
            std = log_std.exp()
            dist = Normal(mean, std)
            old_log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # Compute GAE (Generalized Advantage Estimation)
        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        
        # Compute TD errors (deltas)
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        
        # Compute GAE advantages using reverse cumulative sum
        gae = 0.0
        for t in reversed(range(T)):
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
            mean, log_std = self.actor(states)
            std = log_std.exp()
            dist = Normal(mean, std)
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
            
            # Critic loss
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
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.steps_done = checkpoint.get('steps_done', 0)
