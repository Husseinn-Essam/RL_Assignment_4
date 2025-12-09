"""
SAC, PPO, and TD3 Implementation
This module implements Soft Actor-Critic (SAC), Proximal Policy Optimization (PPO),
and Twin Delayed DDPG (TD3) algorithms for reinforcement learning.
Corrected for Mini-batch PPO updates and TD3 stability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np
import random
from collections import deque, namedtuple

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Experience Replay Buffer for storing and sampling transitions."""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor(np.array([e.state for e in experiences]))
        actions = torch.FloatTensor(np.array([e.action for e in experiences]))
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences])).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences]))
        dones = torch.FloatTensor(np.array([e.done for e in experiences])).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class ActorNetwork(nn.Module):
    """Actor Network for policy-based methods."""
    
    def __init__(self, state_size, action_size, hidden_sizes=[256, 256], discrete=True, use_cnn=False, input_channels=4):
        super(ActorNetwork, self).__init__()
        self.discrete = discrete
        self.use_cnn = use_cnn
        
        if use_cnn:
            self.cnn = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
            with torch.no_grad():
                sample_input = torch.zeros(1, input_channels, 84, 84)
                cnn_output_size = self.cnn(sample_input).shape[1]
            input_size = cnn_output_size
        else:
            input_size = state_size
        
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        self.base = nn.Sequential(*layers)
        
        if discrete:
            self.action_head = nn.Linear(input_size, action_size)
        else:
            self.mean_head = nn.Linear(input_size, action_size)
            self.log_std = nn.Parameter(torch.zeros(action_size))
    
    def forward(self, x):
        if self.use_cnn:
            x = self.cnn(x)
        x = self.base(x)
        
        if self.discrete:
            return self.action_head(x)
        else:
            mean = torch.tanh(self.mean_head(x))
            log_std = self.log_std.expand_as(mean)
            log_std = torch.clamp(log_std, -20, 2) # Standard range for log_std
            return mean, log_std

class CriticNetwork(nn.Module):
    """Critic Network for value function estimation."""
    
    def __init__(self, state_size, action_size=None, hidden_sizes=[256, 256], output_type='value', use_cnn=False, input_channels=4):
        super(CriticNetwork, self).__init__()
        self.output_type = output_type
        self.use_cnn = use_cnn
        
        if use_cnn:
            self.cnn = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
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
        
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state, action=None):
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
    """Soft Actor-Critic (SAC) Agent."""
    
    def __init__(self, state_size, action_size, learning_rate=0.0003, gamma=0.99, tau=0.005, alpha=0.2, 
                 buffer_size=1000000, batch_size=256, hidden_sizes=[256, 256], device='cpu', 
                 discrete=True, use_cnn=False, input_channels=4, reward_scale=1.0, **kwargs):
        
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
        
        self.actor = ActorNetwork(state_size, action_size, hidden_sizes, discrete, use_cnn, input_channels).to(device)
        self.critic1 = CriticNetwork(state_size, action_size, hidden_sizes, 'q_value', use_cnn, input_channels).to(device)
        self.critic2 = CriticNetwork(state_size, action_size, hidden_sizes, 'q_value', use_cnn, input_channels).to(device)
        self.target_critic1 = CriticNetwork(state_size, action_size, hidden_sizes, 'q_value', use_cnn, input_channels).to(device)
        self.target_critic2 = CriticNetwork(state_size, action_size, hidden_sizes, 'q_value', use_cnn, input_channels).to(device)
        
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)
        
        self.memory = ReplayBuffer(buffer_size)
        self.steps_done = 0
    
    def select_action(self, state, epsilon=None):
        if self.use_cnn:
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
                action = torch.tanh(u)
                return action.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward * self.reward_scale, next_state, done)
        self.steps_done += 1
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        if self.use_cnn:
            states = states.float() / 255.0
            next_states = next_states.float() / 255.0
        
        if self.discrete:
            actions = actions.long()
            actions_one_hot = F.one_hot(actions, self.action_size).float()
        else:
            actions_one_hot = actions
            if actions.dim() == 1: actions_one_hot = actions.unsqueeze(1)
        
        with torch.no_grad():
            if self.discrete:
                next_action_probs = self.actor(next_states)
                next_action_probs = F.softmax(next_action_probs, dim=-1)
                next_log_probs = torch.log(next_action_probs + 1e-8)
                
                next_q1_all = torch.zeros(next_states.size(0), self.action_size).to(self.device)
                next_q2_all = torch.zeros(next_states.size(0), self.action_size).to(self.device)
                
                for a in range(self.action_size):
                    a_idx = torch.full((next_states.size(0),), a, device=self.device, dtype=torch.long)
                    action_one_hot = F.one_hot(a_idx, self.action_size).float()
                    next_q1_all[:, a] = self.target_critic1(next_states, action_one_hot).squeeze()
                    next_q2_all[:, a] = self.target_critic2(next_states, action_one_hot).squeeze()
                
                next_q = torch.min(next_q1_all, next_q2_all)
                next_v = (next_action_probs * (next_q - self.alpha * next_log_probs)).sum(dim=1).unsqueeze(1)
                target_q = rewards + (1 - dones) * self.gamma * next_v
            else:
                next_mean, next_log_std = self.actor(next_states)
                next_std = next_log_std.exp()
                next_dist = Normal(next_mean, next_std)
                u_next = next_dist.rsample()
                a_next = torch.tanh(u_next)
                log_pi_next = next_dist.log_prob(u_next).sum(dim=-1, keepdim=True) - \
                              torch.log(1 - torch.tanh(u_next).pow(2) + 1e-6).sum(dim=-1, keepdim=True)
                
                next_q1 = self.target_critic1(next_states, a_next)
                next_q2 = self.target_critic2(next_states, a_next)
                next_q = torch.min(next_q1, next_q2) - self.alpha * log_pi_next
                target_q = rewards + (1 - dones) * self.gamma * next_q
        
        current_q1 = self.critic1(states, actions_one_hot)
        current_q2 = self.critic2(states, actions_one_hot)
        
        critic1_loss = F.smooth_l1_loss(current_q1, target_q)
        critic2_loss = F.smooth_l1_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        if self.discrete:
            logits = self.actor(states)
            action_probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            
            q1_all = torch.zeros(states.size(0), self.action_size).to(self.device)
            q2_all = torch.zeros(states.size(0), self.action_size).to(self.device)
            
            for a in range(self.action_size):
                a_idx = torch.full((states.size(0),), a, device=self.device, dtype=torch.long)
                action_one_hot = F.one_hot(a_idx, self.action_size).float()
                q1_all[:, a] = self.critic1(states, action_one_hot).squeeze()
                q2_all[:, a] = self.critic2(states, action_one_hot).squeeze()
            
            min_q_all = torch.min(q1_all, q2_all)
            actor_loss = (action_probs * (self.alpha * log_probs - min_q_all)).sum(dim=1).mean()
        else:
            mean, log_std = self.actor(states)
            std = log_std.exp()
            dist = Normal(mean, std)
            u = dist.rsample()
            a = torch.tanh(u)
            log_pi = dist.log_prob(u).sum(dim=-1, keepdim=True) - \
                     torch.log(1 - torch.tanh(u).pow(2) + 1e-6).sum(dim=-1, keepdim=True)
            
            q1 = self.critic1(states, a)
            q2 = self.critic2(states, a)
            min_q = torch.min(q1, q2)
            actor_loss = (self.alpha * log_pi - min_q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return (critic1_loss.item() + critic2_loss.item() + actor_loss.item()) / 3

    def get_entropy_coef(self):
        return self.alpha
    
    def save(self, filepath):
        torch.save({
            'algorithm': 'SAC',
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'steps_done': self.steps_done
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        if checkpoint.get('algorithm') not in ['SAC', None]:
            raise ValueError(f"Cannot load {checkpoint.get('algorithm')} model into SAC agent. Model file: {filepath}")
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        self.steps_done = checkpoint.get('steps_done', 0)


class PPOAgent:
    """Proximal Policy Optimization (PPO) Agent with Mini-batch Updates."""
    
    def __init__(self, state_size, action_size, learning_rate=0.0003, gamma=0.99, gae_lambda=0.95, 
                 epsilon_clip=0.2, epochs=10, batch_size=64, hidden_sizes=[256, 256], device='cpu', 
                 discrete=True, entropy_coef=0.01, entropy_decay_episodes=500, max_grad_norm=0.5, 
                 action_scale=1.0, reward_scale=1.0, use_cnn=False, input_channels=4, **kwargs):
        
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
        self.entropy_decay_rate = entropy_coef / max(1, entropy_decay_episodes)
        self.max_grad_norm = max_grad_norm
        self.action_scale = action_scale
        self.reward_scale = reward_scale
        self.use_cnn = use_cnn
        
        self.actor = ActorNetwork(state_size, action_size, hidden_sizes, discrete, use_cnn, input_channels).to(device)
        self.critic = CriticNetwork(state_size, hidden_sizes=hidden_sizes, output_type='value', use_cnn=use_cnn, input_channels=input_channels).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.steps_done = 0
        self.episode_count = 0
    
    def select_action(self, state, epsilon=None):
        if self.use_cnn:
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
                # Store unscaled action for training stability, scale only for environment
                return action.cpu().numpy().flatten()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.states.append(state)
        # For continuous, we store the raw output from the network (before scaling)
        # But in select_action we return raw sample.
        # If the environment wrapper expects scaled, we should scale it THERE.
        # PPO usually trains on the distribution output. 
        # Here we store whatever select_action returned.
        self.actions.append(action)
        self.rewards.append(reward * self.reward_scale)
        self.dones.append(done)
        self.steps_done += 1
    
    def train(self):
        if len(self.states) == 0:
            return None
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)
        
        if self.use_cnn:
            states = states.float() / 255.0
        
        if not self.discrete:
             if actions.dim() == 1:
                actions = actions.unsqueeze(-1)
        else:
             actions = actions.long()

        # Compute GAE and Returns on the full batch (no grad) to ensure correct advantages
        with torch.no_grad():
            values = self.critic(states).squeeze()
            if values.dim() == 0: values = values.unsqueeze(0)
            
            # For simplicity in this implementation, we assume the batch is one trajectory
            # and append a 0 value for the final next_state (or bootstrap if needed)
            # A more robust impl would store next_values or bootstrap properly.
            # Here we assume episode ended or we don't care about last step bootstrap for simplicity
            # (Standard for simple PPO impls, though technically slightly biased at cutoff)
            
            deltas = torch.zeros_like(rewards)
            advantages = torch.zeros_like(rewards)
            
            # We need next values. approximate by shifting values
            next_values = torch.cat([values[1:], torch.tensor([0.0], device=self.device)])
            
            deltas = rewards + self.gamma * next_values * (1 - dones) - values
            
            gae = 0.0
            for t in reversed(range(len(rewards))):
                gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae
            
            returns = advantages + values
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Compute old log probs
            if self.discrete:
                logits = self.actor(states) # Note: For massive CNN batches this might still be heavy
                # If OOM here: chunk this calculation or just store log_probs in buffer
                dist = Categorical(logits=logits)
                old_log_probs = dist.log_prob(actions)
            else:
                mean, log_std = self.actor(states)
                std = log_std.exp()
                dist = Normal(mean, std)
                old_log_probs = dist.log_prob(actions).sum(dim=-1)

        # Mini-batch Updates
        total_actor_loss = 0
        total_critic_loss = 0
        n_updates = 0
        
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                
                # Get mini-batch
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_returns = returns[idx]
                mb_advantages = advantages[idx]
                mb_old_log_probs = old_log_probs[idx]
                
                # Forward pass
                if self.discrete:
                    logits = self.actor(mb_states)
                    dist = Categorical(logits=logits)
                    new_log_probs = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()
                else:
                    mean, log_std = self.actor(mb_states)
                    std = log_std.exp()
                    dist = Normal(mean, std)
                    new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                
                new_values = self.critic(mb_states).squeeze()
                if new_values.dim() == 0: new_values = new_values.unsqueeze(0)
                
                # Ratios
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                # Actor Loss
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                # Critic Loss
                critic_loss = F.smooth_l1_loss(new_values, mb_returns)
                
                # Update
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                n_updates += 1
        
        # Entropy decay
        self.episode_count += 1
        if self.episode_count < self.entropy_decay_episodes:
            self.entropy_coef = self.initial_entropy_coef - (self.entropy_decay_rate * self.episode_count)
        else:
            self.entropy_coef = 0.0
        
        # Clear storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        
        return (total_actor_loss + total_critic_loss) / n_updates if n_updates > 0 else 0

    def get_entropy_coef(self):
        return self.entropy_coef
    
    def save(self, filepath):
        torch.save({
            'algorithm': 'PPO',
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'steps_done': self.steps_done
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        if checkpoint.get('algorithm') not in ['PPO', None]:
            raise ValueError(f"Cannot load {checkpoint.get('algorithm')} model into PPO agent. Model file: {filepath}")
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.steps_done = checkpoint.get('steps_done', 0)


class TD3Agent:
    """Twin Delayed DDPG (TD3) Agent."""
    
    def __init__(self, state_size, action_size, learning_rate=0.0003, gamma=0.99, tau=0.005, 
                 buffer_size=1000000, batch_size=256, hidden_sizes=[256, 256], device='cpu', 
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2, exploration_noise=0.1, 
                 use_cnn=False, input_channels=4, reward_scale=1.0, **kwargs):
        
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
        
        # TD3 Actor is deterministic, we pass discrete=False but only use mean
        self.actor = ActorNetwork(state_size, action_size, hidden_sizes, discrete=False, use_cnn=use_cnn, input_channels=input_channels).to(device)
        self.actor_target = ActorNetwork(state_size, action_size, hidden_sizes, discrete=False, use_cnn=use_cnn, input_channels=input_channels).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic1 = CriticNetwork(state_size, action_size, hidden_sizes, 'q_value', use_cnn=use_cnn, input_channels=input_channels).to(device)
        self.critic2 = CriticNetwork(state_size, action_size, hidden_sizes, 'q_value', use_cnn=use_cnn, input_channels=input_channels).to(device)
        self.target_critic1 = CriticNetwork(state_size, action_size, hidden_sizes, 'q_value', use_cnn=use_cnn, input_channels=input_channels).to(device)
        self.target_critic2 = CriticNetwork(state_size, action_size, hidden_sizes, 'q_value', use_cnn=use_cnn, input_channels=input_channels).to(device)
        
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)
        
        self.memory = ReplayBuffer(buffer_size)
        self.steps_done = 0
        self.total_it = 0
    
    def select_action(self, state, epsilon=None):
        if self.use_cnn:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) / 255.0
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mean, _ = self.actor(state_tensor)
            action = mean.cpu().numpy()[0]
            
            if epsilon is None or epsilon > 0:
                noise = np.random.normal(0, self.exploration_noise, size=self.action_size)
                action = action + noise
                action = np.clip(action, -1.0, 1.0)
            
            return action
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward * self.reward_scale, next_state, done)
        self.steps_done += 1
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return None
        
        self.total_it += 1
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        if self.use_cnn:
            states = states.float() / 255.0
            next_states = next_states.float() / 255.0
            
        # Ensure actions are shaped correctly for Critic (B, ActionDim)
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        
        with torch.no_grad():
            next_mean, _ = self.actor_target(next_states)
            noise = torch.randn_like(next_mean) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_action = next_mean + noise
            next_action = torch.clamp(next_action, -1.0, 1.0)
            
            target_q1 = self.target_critic1(next_states, next_action)
            target_q2 = self.target_critic2(next_states, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()
        
        actor_loss = torch.tensor(0.0)
        if self.total_it % self.policy_delay == 0:
            mean, _ = self.actor(states)
            actor_loss = -self.critic1(states, mean).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return (critic1_loss.item() + critic2_loss.item() + actor_loss.item()) / 3

    def get_entropy_coef(self):
        return self.exploration_noise
    
    def save(self, filepath):
        torch.save({
            'algorithm': 'TD3',
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'steps_done': self.steps_done
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        if checkpoint.get('algorithm') not in ['TD3', None]:
            raise ValueError(f"Cannot load {checkpoint.get('algorithm')} model into TD3 agent. Model file: {filepath}")
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        self.steps_done = checkpoint.get('steps_done', 0)