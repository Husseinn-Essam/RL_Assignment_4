"""
Replay Buffers for Off-Policy RL Agents.
Contains standard and prioritized experience replay implementations.
"""

import torch
import numpy as np
import random
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


class PrioritizedReplayBuffer:
    """
    Proportional Prioritized Experience Replay (PER).
    
    Samples experiences based on their TD-error priority, which helps
    the agent learn more from surprising transitions.
    """
    
    def __init__(self, capacity, alpha=0.6, epsilon=1e-6):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            epsilon: Small constant to ensure non-zero priorities
        """
        self.capacity = int(capacity)
        self.alpha = alpha
        self.epsilon = float(epsilon)

        self.buffer = [None] * self.capacity
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        """Add an experience to the buffer with max priority."""
        self.buffer[self.pos] = Experience(state, action, reward, next_state, done)
        # Initialize new priority to max existing priority, or 1.0 if empty
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of experiences based on priorities.
        
        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            
        Returns:
            Tuple of (states, next_states, actions, rewards, dones, indices, weights)
        """
        if self.size == 0:
            raise ValueError("Trying to sample from an empty buffer")

        prios = self.priorities[:self.size].astype(np.float64) + self.epsilon
        probs = prios ** self.alpha
        probs = probs / probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]

        states = torch.FloatTensor(np.array([e.state for e in experiences]))
        actions = torch.FloatTensor(np.array([e.action for e in experiences]))
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences]))
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences]))
        dones = torch.FloatTensor(np.array([e.done for e in experiences]))

        # Importance-sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        # Normalize by max weight to keep losses stable
        weights = weights / (weights.max() + 1e-8)
        weights = weights.astype(np.float32).reshape(-1, 1)

        return states, next_states, actions, rewards, dones, indices, weights

    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled experiences.
        
        Args:
            indices: Indices of experiences to update
            priorities: New priorities (typically absolute TD-errors)
        """
        for idx, prio in zip(indices, priorities):
            self.priorities[int(idx)] = float(abs(prio)) + self.epsilon

    def __len__(self):
        """Return the current size of the buffer."""
        return self.size
