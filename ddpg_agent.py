import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# ==============================
# Ornstein-Uhlenbeck Noise
# ==============================

class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale

# ==============================
# Actor-Critic Networks
# ==============================

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):  # Tăng hidden_dim
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Thay BatchNorm bằng LayerNorm
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, action_dim),
            nn.Tanh()
        )
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Separate processing for state
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Combined processing
        self.combined_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        state_features = self.state_net(state)
        combined = torch.cat([state_features, action], dim=1)
        return self.combined_net(combined)

# ==============================
# DDPG Agent
# ==============================

class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_bounds, 
                 actor_lr=3e-4, critic_lr=1e-3, tau=0.001, gamma=0.99,
                 batch_size=256, buffer_size=1000000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.action_bounds = action_bounds  # [[min1, max1], [min2, max2]]
        self.batch_size = batch_size
        
        # Networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.target_actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.target_critic = Critic(state_dim, action_dim).to(self.device)

        # Optimizers with weight decay
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=1e-4)

        # Experience replay
        self.memory = deque(maxlen=buffer_size)
        self.tau = tau
        self.gamma = gamma
        
        # Noise for exploration
        self.ou_noise = OUNoise(action_dim)
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.1
        
        # State normalization
        self.state_mean = np.zeros(state_dim)
        self.state_std = np.ones(state_dim)
        self.state_buffer = deque(maxlen=10000)

        # Initialize target networks
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)

    def normalize_state(self, state):
        """Normalize state using running statistics"""
        self.state_buffer.append(state.copy())
        if len(self.state_buffer) > 100:
            states = np.array(list(self.state_buffer))
            self.state_mean = states.mean(axis=0)
            self.state_std = states.std(axis=0) + 1e-8
        
        return (state - self.state_mean) / self.state_std

    def hard_update(self, target, source):
        target.load_state_dict(source.state_dict())

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def remember(self, state, action, reward, next_state, done):
        # Store normalized states
        norm_state = self.normalize_state(state)
        norm_next_state = self.normalize_state(next_state)
        self.memory.append((norm_state, action, reward, norm_next_state, float(done)))

    def select_action(self, state, add_noise=True, training=True):
        norm_state = self.normalize_state(state)
        state_tensor = torch.FloatTensor(norm_state).unsqueeze(0).to(self.device)
        
        # Set networks to appropriate mode
        self.actor.eval()
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        
        if add_noise and training:
            if np.random.rand() < self.epsilon:
                # Random action with higher probability early in training
                action = np.random.uniform(-1, 1, self.action_dim)
            else:
                # Add OU noise for better exploration
                noise = self.ou_noise.noise()
                # Thêm adaptive noise scaling
                noise_scale = max(0.1, self.epsilon)  # Minimum noise
                action = action + noise * noise_scale
        
        # Clamp to [-1, 1] and scale
        action = np.clip(action, -1, 1)
        return self.scale_action(action)

    def scale_action(self, action):
        """Map [-1, 1] -> [min, max]"""
        scaled_action = []
        for i in range(self.action_dim):
            min_val, max_val = self.action_bounds[i]
            scaled_val = min_val + (action[i] + 1) * 0.5 * (max_val - min_val)
            scaled_action.append(scaled_val)
        return np.array(scaled_action)
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Set networks to training mode
        self.actor.train()
        self.critic.train()
        self.target_actor.eval()
        self.target_critic.eval()

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Critic loss
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            target_value = rewards + self.gamma * target_q * (1 - dones)

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_value)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Actor loss - only update every 2 steps for stability
        if hasattr(self, 'update_counter'):
            self.update_counter += 1
        else:
            self.update_counter = 0
            
        if self.update_counter % 2 == 0:
            predicted_actions = self.actor(states)
            actor_loss = -self.critic(states, predicted_actions).mean()

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # Soft update target networks
            self.soft_update(self.target_actor, self.actor)
            self.soft_update(self.target_critic, self.critic)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Reset noise occasionally
        if self.update_counter % 1000 == 0:
            self.ou_noise.reset()

    def save_model(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_mean': self.state_mean,
            'state_std': self.state_std
        }, filepath)

    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.state_mean = checkpoint.get('state_mean', self.state_mean)
        self.state_std = checkpoint.get('state_std', self.state_std)