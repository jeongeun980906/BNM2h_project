import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self,hidden_size=256,action_size=4,log_std_min=-20, log_std_max=2):
        super(Actor,self).__init__()
        self.log_std_min=log_std_min
        self.log_std_max=log_std_max
        self.conv1=nn.Conv2d(4,hidden_size,kernel_size=(3,3),padding=(1,1))
        self.conv2=nn.Conv2d(hidden_size,hidden_size,kernel_size=(3,3),padding=(1,1))
        self.conv3=nn.Conv2d(hidden_size,16,kernel_size=(3,3))
        self.pooling=nn.MaxPool2d(2)
        self.fc1=nn.Linear(16*48*48,128)
        self.fc2=nn.Linear(128,action_size)
        self.fc3=nn.Linear(128,action_size)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pooling(x)
        x=F.relu(self.conv2(x))
        x=self.pooling(x)
        x=F.relu(self.conv3(x))
        x=x.view(-1,16*48*48)
        x=F.relu(self.fc1(x))
        mu=self.fc2(x)
        log_std=torch.clamp(self.fc3(x),self.log_std_min,self.log_std_max)
        return mu,log_std

class Critic(nn.Module):
    def __init__(self,hidden_size=256,action_size=4,log_std_min=-20, log_std_max=2):
        super(Critic,self).__init__() #Qfunction
        self.conv1=nn.Conv2d(4,hidden_size,kernel_size=(3,3),padding=(1,1))
        self.conv2=nn.Conv2d(hidden_size,hidden_size,kernel_size=(3,3),padding=(1,1))
        self.conv3=nn.Conv2d(hidden_size,16,kernel_size=(3,3))
        self.pooling=nn.MaxPool2d(2)
        self.fc1=nn.Linear(16*48*48,128)
        self.fc2=nn.Linear(128,action_size)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pooling(x)
        x=F.relu(self.conv2(x))
        x=self.pooling(x)
        x=F.relu(self.conv3(x))
        x=x.view(-1,16*48*48)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x