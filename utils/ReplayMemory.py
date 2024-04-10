import random
from collections import namedtuple, deque
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class PrioritizedReplayBuffer:
    def __init__(self, capacity, epsilon=1e-6, alpha=0.8, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.epsilon = epsilon
        self.alpha = alpha   # how much prioritisation is used
        self.beta = beta    # for importance sampling weights
        self.beta_increment = beta_increment
        self.priority_buffer = np.zeros(self.capacity)
        self.data = []
        self.position = 0
    
    def __len__(self):
        return len(self.data)

    def push(self, *args):
        experience = Transition(*args)
        
        max_priority = np.max(self.priority_buffer) if self.data else 1.0
        if len(self.data) < self.capacity:
            self.data.append(experience)
        else:
            self.data[self.position] = experience
        self.priority_buffer[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        priorities = self.priority_buffer[:len(self.data)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.data), batch_size, p=probabilities)
        experiences = [self.data[i] for i in indices]

        total = len(self.data)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = np.min([1., self.beta + self.beta_increment])
        
        return experiences, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priority_buffer[idx] = error + self.epsilon