import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class gym_Actor(nn.Module):
    def __init__(self,state_size=3,hidden_size=128,action_size=1,log_std_min=-20, log_std_max=2):
        super(gym_Actor,self).__init__()
        self.log_std_min=log_std_min
        self.log_std_max=log_std_max
        self.fc1=nn.Linear(state_size,hidden_size)
        self.fc2=nn.Linear(hidden_size,hidden_size)
        self.fc_mu=nn.Linear(hidden_size,action_size)
        self.fc_std=nn.Linear(hidden_size,action_size)
        self.L=nn.LayerNorm(hidden_size)
        
        nn.init.kaiming_normal_(self.fc1.weight.data)
        nn.init.kaiming_normal_(self.fc2.weight.data)
        nn.init.kaiming_normal_(self.fc_std.weight.data)
        nn.init.kaiming_normal_(self.fc_mu.weight.data)

    def forward(self,x):
        x=F.relu(self.L(self.fc1(x)))
        x=F.relu(self.L(self.fc2(x)))
        mu=self.fc_mu(x)
        log_std=torch.clamp(self.fc_std(x),self.log_std_min,self.log_std_max)
        return mu,log_std

class gym_Critic(nn.Module):
    def __init__(self,state_size=3,hidden_size=128,action_size=1,log_std_min=-20, log_std_max=2):
        super(gym_Critic,self).__init__() #Qfunction
        self.fc1=nn.Linear(state_size+action_size,hidden_size)
        self.fc2=nn.Linear(hidden_size,hidden_size)
        self.fc3=nn.Linear(hidden_size,action_size)
        nn.init.kaiming_normal_(self.fc1.weight.data)
        nn.init.kaiming_normal_(self.fc2.weight.data)
        nn.init.kaiming_normal_(self.fc3.weight.data)
        self.L=nn.LayerNorm(hidden_size)

    def forward(self,x,a):
        x=torch.cat([x,a],dim=1)
        x=F.relu(self.L(self.fc1(x)))
        x=F.relu(self.L(self.fc2(x)))
        x=self.fc3(x)
        return x