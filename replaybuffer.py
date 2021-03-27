import numpy as np
from collections import deque,namedtuple
import torch
import random

class memory():
    def __init__(self,action_size=4):
        self.action_size=action_size
        self.buffer=deque(maxlen=10000)
        self.exp=namedtuple('Experience',field_names=['state','action','reward','next_state','done'])
    def store(self,state,action,reward,next_state,done):
        transition=self.exp(state,action,reward,next_state,done)
        self.buffer.append(transition)
    def sample(self,batch_size=4,device=torch.device('cuda')):
        experiences=random.sample(self.buffer,k=batch_size)
        states = torch.from_numpy(np.vstack([[e.state] for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([[e.next_state] for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states,actions,rewards,next_states,dones)