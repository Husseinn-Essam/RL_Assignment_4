"""
Twin Delayed DDPG (TD3) Agent Implementation.
Uses three key improvements over DDPG:
1. Clipped Double Q-learning to reduce overestimation bias
2. Delayed policy updates
3. Target policy smoothing
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from networks import ActorNetwork, CriticNetwork
from buffers import ReplayBuffer, PrioritizedReplayBuffer


class TD3Agent:
    """Twin Delayed DDPG (TD3) Agent for continuous action spaces."""
    
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
        start_steps=10000,
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
            use_cnn: Whether to use CNN for image inputs
            input_channels: Number of input channels for CNN
            reward_scale: Scale factor for rewards
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
        self.actor = ActorNetwork(state_size, action_size, hidden_sizes, use_cnn, input_channels).to(device)
        self.actor_target = ActorNetwork(state_size, action_size, hidden_sizes, use_cnn, input_channels).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Twin Q-networks (critics)
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
        
        # Replay buffer (optionally prioritized)
        self.use_per = kwargs.get('use_per', False)
        self.per_alpha = kwargs.get('per_alpha', 0.6)
        self.per_eps = kwargs.get('per_eps', 1e-6)
        self.per_beta = kwargs.get('per_beta', 0.4)
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(buffer_size, alpha=self.per_alpha, epsilon=self.per_eps)
        else:
            self.memory = ReplayBuffer(buffer_size)
        
        self.steps_done = 0
        self.total_it = 0  # Total training iterations for delayed policy updates
    
    def select_action(self, state, epsilon=None):
        """
        Select action using the deterministic policy with exploration noise.
        
        Args:
            state: Current state
            epsilon: If 0, no exploration noise is added (for testing)
        
        Returns:
            Selected action with exploration noise
        """
        # If we're still in the initial random collection phase, return random actions
        if hasattr(self, 'start_steps') and self.steps_done < self.start_steps:
            return np.random.uniform(-1.0, 1.0, size=self.action_size)

        if self.use_cnn:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) / 255.0
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # TD3 uses deterministic policy (only mean, no log_std)
            mean, _ = self.actor(state_tensor)
            action = mean.cpu().numpy()[0]

            # Add exploration noise during training
            if epsilon is None or epsilon > 0:
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
        
        # 1. SAMPLE BATCH
        if self.use_per:
            states, next_states, actions, rewards, dones, indices, weights = self.memory.sample(self.batch_size, beta=self.per_beta)
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
            weights = 1.0 # Uniform weights if not using PER
        
        # Normalize image inputs
        if self.use_cnn:
            states = states.float() / 255.0
            next_states = next_states.float() / 255.0
        
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        
        # 2. CRITIC UPDATE
        with torch.no_grad():
            next_mean, _ = self.actor_target(next_states)
            noise = torch.randn_like(next_mean) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_action = next_mean + noise
            next_action = torch.clamp(next_action, -1.0, 1.0)
            
            target_q1 = self.target_critic1(next_states, next_action).squeeze()
            target_q2 = self.target_critic2(next_states, next_action).squeeze()
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q1 = self.critic1(states, actions).squeeze()
        current_q2 = self.critic2(states, actions).squeeze()
        
        # [FIX] Apply Importance Sampling Weights (PER)
        # We use reduction='none' to get individual losses, multiply by weights, then mean
        if self.use_per:
            critic1_loss = (F.mse_loss(current_q1, target_q, reduction='none') * weights.squeeze()).mean()
            critic2_loss = (F.mse_loss(current_q2, target_q, reduction='none') * weights.squeeze()).mean()
        else:
            critic1_loss = F.mse_loss(current_q1, target_q)
            critic2_loss = F.mse_loss(current_q2, target_q)
        
        # Update critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        # [FIX] Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()
        
        # Update critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        # [FIX] Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()

        # [FIX] Update Priorities in Buffer
        if self.use_per:
            # TD error is used as priority
            td_error1 = torch.abs(current_q1 - target_q).detach().cpu().numpy()
            td_error2 = torch.abs(current_q2 - target_q).detach().cpu().numpy()
            # Use the mean or max of the two errors
            new_priorities = (td_error1 + td_error2) / 2.0
            self.memory.update_priorities(indices, new_priorities)
        
        # 3. ACTOR UPDATE (Delayed)
        actor_loss = torch.tensor(0.0)
        if self.total_it % self.policy_delay == 0:
            mean, _ = self.actor(states)
            actor_loss = -self.critic1(states, mean).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # [FIX] Clip actor gradients too
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
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
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
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
