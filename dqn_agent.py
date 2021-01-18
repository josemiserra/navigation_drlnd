import numpy as np
import random
from collections import namedtuple, deque
import heapq
from model import DuelingQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64       # minibatch size
GAMMA = 0.99          # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4              # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
eps = np.finfo(np.float32).eps

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, priority = True, beta_start = 0.4, beta_episodes = 400):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.is_priority = priority
        if priority: 
            self.memory = PriorityReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed) 
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.frame_step = 0

        self.beta_start = 0.4
        self.beta_episodes = 200
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)    
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                if self.is_priority:
                    beta = min(1.0, self.beta_start + self.frame_step* (1.0 - self.beta_start) / self.beta_episodes)
                    experiences = self.memory.sample(beta)
                else:
                    experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
        if done:
            self.frame_step += 1

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
            

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        if self.is_priority:
            states, actions, rewards, next_states, dones, weights, indices = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        ## Double DQN 
        #use qnetwork_local to find the best action. In function max, 0 returns value, 1 returns index
        best_actions_ind = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).gather(1,best_actions_ind)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        if self.is_priority:
            loss = (((Q_targets - Q_expected).pow(2)).squeeze())*weights
            # clamp btwn -1..1
            # loss = torch.clamp(loss, -1., 1.)
            deltas = loss + eps
            loss = loss.mean()
        else:
            loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)  
        if self.is_priority:
            self.memory.update_priorities(indices, deltas)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


DEFAULT_PRIORITY = 1e-10 # how much is the minimum priority given to each experience

class PriorityReplayBuffer(ReplayBuffer):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha = 0.6):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["priority","tie_breaker","state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.time_point = 0
        self.alpha = alpha
        self.default_priority = DEFAULT_PRIORITY**self.alpha
        self.max_priority = self.default_priority

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(self.max_priority, self.time_point, state, action, reward, next_state, done)
        self.time_point += 1
        self.memory.append(e)

    def sample(self, beta = 0.4):
        """Randomly sample a batch of experiences from memory."""
        
        priorities = [e.priority for e in self.memory]
        total = len(priorities)
        sampling_probs = np.array(priorities)
        sampling_probs /= sampling_probs.sum()
        weights = (total * sampling_probs) ** (-beta)
        weights /= weights.max()

        indices = np.random.choice(range(len(self.memory)), size=self.batch_size, replace=False, p=sampling_probs)
        experiences = [self.memory[ind] for ind in indices]

        weights = torch.from_numpy(np.array(weights[indices], dtype=np.float32)).float().to(device)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones, weights, indices)

    def update_priorities(self, experience_indices, errors):
        # update of the priorities
        for exp, error in zip(experience_indices, errors.cpu().data.numpy()):
            self.memory[exp] = self.memory[exp]._replace(priority=np.abs(error.item())**self.alpha)
            self.max_priority = max(self.memory[exp].priority, self.max_priority)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)