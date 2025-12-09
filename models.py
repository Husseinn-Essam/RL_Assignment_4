"""
SAC, PPO, and TD3 Implementation (updated)
Key fixes for PPO on CarRacing:
 - Proper tanh-squashed Gaussian sampling with log-prob correction
 - Stronger CNN encoder for pixel inputs
 - PPO stores sampled (squashed) action and the precomputed log_prob per step
 - PPO value and advantage computation adjusted for stability
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np
import random
from collections import deque, namedtuple

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor(np.array([e.state for e in experiences]))
        actions = torch.FloatTensor(np.array([e.action for e in experiences]))
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences])).unsqueeze(-1)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences]))
        dones = torch.FloatTensor(np.array([e.done for e in experiences])).unsqueeze(-1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ---------- CNN Encoder (shared style) ----------
class ConvEncoder(nn.Module):
    """
    Lightweight but deeper conv encoder suitable for CarRacing pixel input.
    Input: (B, C=4, 84, 84)
    Output: flattened features vector
    """

    def __init__(self, in_channels=4, feature_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),  # -> 20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),           # -> 9x9
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),          # -> 7x7
            nn.ReLU(),
            nn.Flatten()
        )
        # compute flatten size dynamically
        with torch.no_grad():
            t = torch.zeros(1, in_channels, 84, 84)
            flat = self.conv(t)
            flat_size = flat.shape[1]
        self.fc = nn.Sequential(
            nn.Linear(flat_size, feature_dim),
            nn.ReLU()
        )
        self._out_dim = feature_dim

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

    @property
    def out_dim(self):
        return self._out_dim


# ---------- Actor Network ----------
class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[256, 256], discrete=True, use_cnn=False, input_channels=4):
        super(ActorNetwork, self).__init__()
        self.discrete = discrete
        self.use_cnn = use_cnn

        if use_cnn:
            self.encoder = ConvEncoder(in_channels=input_channels, feature_dim=512)
            input_size = self.encoder.out_dim
        else:
            input_size = state_size

        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(input_size, h))
            layers.append(nn.ReLU())
            input_size = h
        self.base = nn.Sequential(*layers)

        if discrete:
            self.action_head = nn.Linear(input_size, action_size)
        else:
            self.mean_head = nn.Linear(input_size, action_size)
            # state-independent log_std parameter (learnable)
            self.log_std = nn.Parameter(torch.ones(action_size) * -0.5)

    def forward(self, x):
        if self.use_cnn:
            x = x.float()
            x = self.encoder(x)
        x = self.base(x)
        if self.discrete:
            logits = self.action_head(x)
            return logits
        else:
            # Return raw mean (NOT tanh) — we'll apply tanh when sampling
            mean = self.mean_head(x)
            log_std = self.log_std.expand_as(mean)
            log_std = torch.clamp(log_std, -20, 2)  # stable range
            return mean, log_std


# ---------- Critic Network ----------
class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size=None, hidden_sizes=[256, 256], output_type='value', use_cnn=False, input_channels=4):
        super(CriticNetwork, self).__init__()
        self.output_type = output_type
        self.use_cnn = use_cnn

        if use_cnn:
            self.encoder = ConvEncoder(in_channels=input_channels, feature_dim=512)
            if output_type == 'q_value' and action_size is not None:
                input_size = self.encoder.out_dim + action_size
            else:
                input_size = self.encoder.out_dim
        else:
            if output_type == 'q_value' and action_size is not None:
                input_size = state_size + action_size
            else:
                input_size = state_size

        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(input_size, h))
            layers.append(nn.ReLU())
            input_size = h
        layers.append(nn.Linear(input_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state, action=None):
        if self.use_cnn:
            sf = self.encoder(state)
            if self.output_type == 'q_value' and action is not None:
                x = torch.cat([sf, action], dim=-1)
            else:
                x = sf
        else:
            if self.output_type == 'q_value' and action is not None:
                x = torch.cat([state, action], dim=-1)
            else:
                x = state
        return self.network(x)


# ---------- SAC Agent (unchanged core but consistent use_cnn) ----------
class SACAgent:
    def __init__(self, state_size, action_size, learning_rate=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 buffer_size=1000000, batch_size=256, hidden_sizes=[256, 256], device='cpu',
                 discrete=False, use_cnn=False, input_channels=4, reward_scale=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.device = device
        self.use_cnn = use_cnn
        self.reward_scale = reward_scale
        self.discrete = discrete

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
                a = torch.tanh(u)
                return a.cpu().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward * self.reward_scale, next_state, done)
        self.steps_done += 1

    def train(self):
        if len(self.memory) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device).squeeze(-1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).squeeze(-1)

        if self.use_cnn:
            states = states / 255.0
            next_states = next_states / 255.0

        with torch.no_grad():
            next_mean, next_log_std = self.actor(next_states)
            next_std = next_log_std.exp()
            next_dist = Normal(next_mean, next_std)
            u_next = next_dist.rsample()
            a_next = torch.tanh(u_next)
            log_pi_next = next_dist.log_prob(u_next).sum(dim=-1) - torch.log(1 - a_next.pow(2) + 1e-7).sum(dim=-1)
            q1_next = self.target_critic1(next_states, a_next).squeeze()
            q2_next = self.target_critic2(next_states, a_next).squeeze()
            next_q = torch.min(q1_next, q2_next) - self.alpha * log_pi_next
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # ensure actions shape matches
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)

        current_q1 = self.critic1(states, actions).squeeze()
        current_q2 = self.critic2(states, actions).squeeze()

        critic1_loss = F.mse_loss(current_q1, target_q.detach())
        critic2_loss = F.mse_loss(current_q2, target_q.detach())

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # actor
        mean, log_std = self.actor(states)
        std = log_std.exp()
        dist = Normal(mean, std)
        u = dist.rsample()
        a = torch.tanh(u)
        log_pi = dist.log_prob(u).sum(dim=-1) - torch.log(1 - a.pow(2) + 1e-7).sum(dim=-1)
        q1 = self.critic1(states, a).squeeze()
        q2 = self.critic2(states, a).squeeze()
        min_q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_pi - min_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # soft update targets
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return (critic1_loss.item() + critic2_loss.item() + actor_loss.item()) / 3

    def get_entropy_coef(self):
        return self.alpha

    def save(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'steps_done': self.steps_done
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        self.steps_done = checkpoint.get('steps_done', 0)


# ---------- PPO Agent (FIXED: tanh-squashed Gaussian + log_prob correction + stronger CNN) ----------
class PPOAgent:
    def __init__(self,
                 state_size,
                 action_size,
                 learning_rate=3e-4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 epsilon_clip=0.2,
                 epochs=10,
                 batch_size=64,
                 hidden_sizes=[256, 256],
                 device='cpu',
                 discrete=False,
                 entropy_coef=0.05,
                 entropy_decay_episodes=500,
                 max_grad_norm=0.5,
                 action_scale=1.0,
                 reward_scale=0.1,
                 use_cnn=True,
                 input_channels=4):
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
        self.reward_scale = reward_scale if not discrete else 1.0
        self.episode_count = 0
        self.use_cnn = use_cnn

        self.actor = ActorNetwork(state_size, action_size, hidden_sizes, discrete, use_cnn, input_channels).to(device)
        self.critic = CriticNetwork(state_size, hidden_sizes=hidden_sizes, output_type='value', use_cnn=use_cnn, input_channels=input_channels).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # trajectory buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        # store log_probs from sampling (with tanh correction)
        self.actions_log_probs = []

        # for select_action -> store last computed log_prob so store_transition can pick it up
        self.last_log_prob = None

        self.steps_done = 0

    def select_action(self, state, epsilon=None):
        """
        Return an action in [-1,1]. For continuous actions we sample u ~ N(mean,std), compute a = tanh(u),
        compute log_prob with tanh-change-of-variable correction and cache it in self.last_log_prob.
        """
        if self.use_cnn:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) / 255.0
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.discrete:
                logits = self.actor(state_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample()
                self.last_log_prob = dist.log_prob(action).cpu().numpy()
                return int(action.item())
            else:
                mean, log_std = self.actor(state_tensor)  # mean: raw (unbounded)
                std = log_std.exp()
                dist = Normal(mean, std)
                u = dist.rsample()               # pre-squash
                a = torch.tanh(u)                # squashed action in (-1,1)
                # log_prob with tanh correction
                log_prob = dist.log_prob(u).sum(dim=-1) - torch.log(1 - a.pow(2) + 1e-7).sum(dim=-1)
                self.last_log_prob = log_prob.cpu().numpy().astype(np.float32)
                a_np = a.cpu().numpy().flatten()
                # clipping to be safe but not necessary
                a_np = np.clip(a_np, -1.0, 1.0)
                return a_np

    def store_transition(self, state, action, reward, next_state, done):
        """Store the transition for PPO. If select_action computed last_log_prob, store it as well."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward * self.reward_scale)
        self.next_states.append(next_state)
        self.dones.append(done)
        # If last_log_prob exists, use it; else push a placeholder (0). This is robust if env resets happen.
        if self.last_log_prob is not None:
            # last_log_prob may be array; we want scalar per-sample logprob
            if isinstance(self.last_log_prob, np.ndarray) and self.last_log_prob.size > 1:
                # store scalar sum if multi-dim (should already be summed in select_action)
                self.actions_log_probs.append(float(np.sum(self.last_log_prob)))
            else:
                self.actions_log_probs.append(float(self.last_log_prob))
            # reset
            self.last_log_prob = None
        else:
            self.actions_log_probs.append(0.0)
        self.steps_done += 1

    def train(self):
        """Run PPO update on collected trajectory (on-policy)."""
        if len(self.states) == 0:
            return None

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.actions_log_probs)).to(self.device)

        if self.use_cnn:
            states = states / 255.0
            next_states = next_states / 255.0

        if self.discrete:
            actions = torch.LongTensor(self.actions).to(self.device)
        else:
            actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
            if actions.dim() == 1:
                actions = actions.unsqueeze(-1)

        # Compute values and next_values (for GAE)
        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            if values.dim() == 0:
                values = values.unsqueeze(0)
            if next_values.dim() == 0:
                next_values = next_values.unsqueeze(0)

        T = len(rewards)
        deltas = rewards + self.gamma * next_values * (1 - dones) - values

        advantages = torch.zeros(T, device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for _ in range(self.epochs):
            # New policy log probs (must use same tanh-correction formula)
            if self.discrete:
                logits = self.actor(states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
            else:
                mean, log_std = self.actor(states)
                std = log_std.exp()
                dist = Normal(mean, std)

                # actions are the squashed actions stored earlier (in [-1,1]).
                # To compute new_log_probs we need to invert tanh (atanh) to get u:
                # u = atanh(a) = 0.5 * ln((1+a)/(1-a)). Numerically stable handling required.
                a = actions
                # clamp a inside (-1+eps, 1-eps)
                eps = 1e-6
                a_clamped = torch.clamp(a, -1 + eps, 1 - eps)
                u = 0.5 * (torch.log1p(a_clamped) - torch.log1p(-a_clamped))  # atanh

                # compute log_prob of u under Gaussian then subtract tanh correction
                log_prob_u = dist.log_prob(u).sum(dim=-1)
                log_prob = log_prob_u - torch.log(1 - a_clamped.pow(2) + 1e-7).sum(dim=-1)
                new_log_probs = log_prob
                entropy = dist.entropy().sum(dim=-1).mean()

            new_values = self.critic(states).squeeze()
            if new_values.dim() == 0:
                new_values = new_values.unsqueeze(0)

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            critic_loss = F.mse_loss(new_values, returns.detach())

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

        # decay entropy
        self.episode_count += 1
        if self.episode_count < self.entropy_decay_episodes:
            self.entropy_coef = self.initial_entropy_coef - (self.entropy_decay_rate * self.episode_count)
        else:
            self.entropy_coef = 0.0

        avg_loss = (total_actor_loss + total_critic_loss) / (2 * self.epochs)

        # Clear buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.actions_log_probs = []

        return avg_loss

    def get_entropy_coef(self):
        return self.entropy_coef

    def save(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'steps_done': self.steps_done
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.steps_done = checkpoint.get('steps_done', 0)


# ---------- TD3 Agent (kept consistent with new encoder) ----------
class TD3Agent:
    def __init__(self, state_size, action_size, learning_rate=3e-4, gamma=0.99, tau=0.005,
                 buffer_size=1000000, batch_size=256, hidden_sizes=[256, 256], device='cpu',
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2, exploration_noise=0.1,
                 use_cnn=False, input_channels=4, reward_scale=1.0):
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
        rewards = rewards.to(self.device).squeeze(-1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).squeeze(-1)

        if self.use_cnn:
            states = states / 255.0
            next_states = next_states / 255.0

        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)

        with torch.no_grad():
            next_mean, _ = self.actor_target(next_states)
            noise = (torch.randn_like(next_mean) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_mean + noise).clamp(-1.0, 1.0)

            target_q1 = self.target_critic1(next_states, next_action).squeeze()
            target_q2 = self.target_critic2(next_states, next_action).squeeze()
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q1 = self.critic1(states, actions).squeeze()
        current_q2 = self.critic2(states, actions).squeeze()

        critic1_loss = F.mse_loss(current_q1, target_q.detach())
        critic2_loss = F.mse_loss(current_q2, target_q.detach())

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
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

        return (critic1_loss.item() + critic2_loss.item() + float(actor_loss)) / 3

    def get_entropy_coef(self):
        return self.exploration_noise

    def save(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'steps_done': self.steps_done,
            'total_it': self.total_it
        }, filepath)

    def load(self, filepath):
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
