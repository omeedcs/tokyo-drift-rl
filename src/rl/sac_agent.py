"""
Soft Actor-Critic (SAC) agent for drift control.

Implements SAC algorithm for continuous control of autonomous drifting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Optional
from collections import deque
import random


class ReplayBuffer:
    """Experience replay buffer for SAC."""
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add transition to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones).unsqueeze(1)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


class Actor(nn.Module):
    """Actor network for SAC (Gaussian policy)."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        """
        Initialize actor network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super(Actor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Network
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Mean and log std outputs
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of (mean, log_std)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of (action, log_prob)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Reparameterization trick
        normal = Normal(mean, std)
        x_t = normal.rsample()
        
        # Apply tanh squashing
        action = torch.tanh(x_t)
        
        # Compute log probability with squashing correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    """Critic network for SAC (Q-function)."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize critic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
        """
        super(Critic, self).__init__()
        
        # Q1 network
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Q2 network (for double Q-learning)
        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both Q-networks.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Tuple of (Q1, Q2) values
        """
        sa = torch.cat([state, action], 1)
        
        # Q1
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        
        # Q2
        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        
        return q1, q2


class SACAgent:
    """
    Soft Actor-Critic agent.
    
    Implements SAC algorithm for continuous control:
    - Maximum entropy RL
    - Off-policy learning
    - Double Q-learning
    - Automatic temperature tuning
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy_tuning: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize SAC agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            lr: Learning rate
            gamma: Discount factor
            tau: Target network update rate
            alpha: Entropy temperature
            auto_entropy_tuning: Whether to automatically tune temperature
            device: Device to run on ("cpu" or "cuda")
        """
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_dim = action_dim
        
        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Copy parameters to target network
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Automatic entropy tuning
        self.auto_entropy_tuning = auto_entropy_tuning
        if auto_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Training stats
        self.total_steps = 0
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_values = []
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """
        Select action from policy.
        
        Args:
            state: Current state
            evaluate: Whether to use deterministic policy (eval mode)
            
        Returns:
            Action array
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if evaluate:
            # Deterministic policy for evaluation
            mean, _ = self.actor.forward(state)
            action = torch.tanh(mean)
        else:
            # Stochastic policy for exploration
            action, _ = self.actor.sample(state)
        
        return action.detach().cpu().numpy()[0]
    
    def update(self, batch_size: int = 256) -> dict:
        """
        Update networks using a batch from replay buffer.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Dictionary of losses
        """
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next
        
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature (alpha)
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # Soft update target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Track stats
        self.total_steps += 1
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.alpha_values.append(self.alpha)
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "alpha": self.alpha
        }
    
    def save(self, filepath: str):
        """Save agent to file."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha if self.auto_entropy_tuning else None,
            "total_steps": self.total_steps
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if self.auto_entropy_tuning and checkpoint["log_alpha"] is not None:
            self.log_alpha = checkpoint["log_alpha"]
        self.total_steps = checkpoint["total_steps"]
