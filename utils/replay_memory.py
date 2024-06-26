import random
from collections import deque
from utils.transition import Transition

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def reset_memory(self):
        self.memory.clear()

    def sample_all(self):
        return self.memory
    def __len__(self):
        return len(self.memory)