import matplotlib.pyplot as plt
import numpy as np
import torch
from net import Actor,Critic
from replaybuffer import memory
from torch.distributions import Normal, MultivariateNormal
import torch.optim as optim
from gym_net import gym_Actor,gym_Critic

class Agent():
    def __init__(self,action_size=4,action_scale=1,state_size=3,device=torch.device('cuda')):
        self.gamma=0.99
        self.alpha=1.0
        self.lr_act=0.001
        self.lr_cri=0.001
        self.action_size=action_size
        self.batch_size=128
        self.tau=0.1
        self.alpha=0.1
        self.action_scale=action_scale

        self.entropy_adjustment=False
        #self.Actor=Actor(action_size=action_size).to(device)
        #self.Critic=Critic(action_size=action_size).to(device)
        self.Actor=gym_Actor(state_size=state_size,hidden_size=128,action_size=action_size).to(device)
        self.Critic1=gym_Critic(state_size=state_size,hidden_size=128,action_size=action_size).to(device)
        self.Critic2=gym_Critic(state_size=state_size,hidden_size=128,action_size=action_size).to(device)
        self.target_critic1=gym_Critic(state_size=state_size,hidden_size=128,action_size=action_size).to(device)
        self.target_critic2=gym_Critic(state_size=state_size,hidden_size=128,action_size=action_size).to(device)
        self.target_critic1.load_state_dict(self.Critic1.state_dict())
        self.target_critic2.load_state_dict(self.Critic2.state_dict())

        self.memory=memory()
        self.moving_avarage=moving_avarage(state_size)
        self.actor_optimizer=optim.Adam(self.Actor.parameters(),lr=self.lr_act)
        self.critic_optimizer1=optim.Adam(self.Critic1.parameters(),lr=self.lr_cri)
        self.critic_optimizer2=optim.Adam(self.Critic2.parameters(),lr=self.lr_cri)
        if self.entropy_adjustment:
            self.lr_alpha=0.0001
            self.target_entropy = -torch.prod(torch.Tensor(self.action_size).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer=optim.Adam([self.log_alpha],lr=self.lr_alpha)
        
        self.device=device
    
    def soft_upate(self,net,target_net):
        for param, target_param in zip(net.parameters(),target_net.parameters()):
            target_param.data.copy_(self.tau*param.data+(1.0-self.tau)*target_param.data)
    
    def sample_action(self,state,batch_size=1):
        mu,log_std=self.Actor(state)
        #print(mu,log_std)
        dst=Normal(0,1)
        temp=dst.sample_n(self.action_size*batch_size).to(self.device)
        temp=temp.view((batch_size,self.action_size))
        log_prob=dst.log_prob(temp)
        action=mu+log_std*temp
        action=torch.tanh(action)*self.action_scale
        #print(action)
        return action,log_prob

    def train(self):
        (states,actions,rewards,next_states,dones)=self.memory.sample(self.batch_size)
        # states=self.moving_avarage.normalize(states,self.batch_size)
        # states[torch.isnan(states)] = 0
        # print('res',states)
        # next_states=self.moving_avarage.normalize(next_states,self.batch_size)
        # next_states[torch.isnan(next_states)] = 0

        next_action, next_log_prob=self.sample_action(next_states,self.batch_size)
        with torch.no_grad():
            next_q1=self.target_critic1(next_states,next_action)
            next_q2=self.target_critic2(next_states,next_action)
            target_q1=rewards+self.gamma*(next_q1-self.alpha*next_log_prob)
            target_q2=rewards+self.gamma*(next_q2-self.alpha*next_log_prob)
        q1=self.Critic1(states,actions)
        q2=self.Critic2(states,actions)

        sampled_action,sampled_log_prob=self.sample_action(states,self.batch_size)
        sq1=self.Critic1(states,sampled_action)
        sq2=self.Critic2(states,sampled_action)
        sq=torch.min(sq1,sq2)

        if self.entropy_adjustment:
            alpha_loss = -(self.log_alpha * (sampled_log_prob + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
        # policy loss
        policy_loss=(self.alpha*sampled_log_prob-sq).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # critic loss
        MSE = torch.nn.MSELoss()
        critic_loss1=MSE(q1,target_q1)
        critic_loss2=MSE(q2,target_q2)

        self.critic_optimizer1.zero_grad()
        critic_loss1.backward()
        self.critic_optimizer1.step()

        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        self.critic_optimizer2.step()
    
    def update_param(self):
        self.soft_upate(self.Critic1,self.target_critic1)
        self.soft_upate(self.Critic2,self.target_critic2)

    def save_nn(self,e,score):
        torch.save(self.Actor,"./saved_model/Actor_{}_{}.pth".format(e,score))

    def load(self,e,score,device=torch.device('cuda')):
        PATH="./saved_model/Actor_{}_{}.pth".format(e,score)
        self.Actor=torch.load(PATH).to(device)
        PATH="./saved_model/Critic_{}_{}.pth".format(e,score)
        self.Critic=torch.load(PATH).to(device)

class moving_avarage():
    def __init__(self,state_size):
        self.state_size=state_size
        self.r_mean=np.zeros((state_size))
        self.r_var=np.zeros((state_size))
        self.cnt=0
    def run(self,state):
        #print('s',state)
        self.cnt+=1
        r_mean_new=self.r_mean+(state-self.r_mean)/self.cnt
        self.r_var+=(state - self.r_mean)*(state - r_mean_new)
        self.r_mean=r_mean_new
        #print(self.r_mean,self.r_var)
    def normalize(self,state,batch_size,device='cuda'):
        if self.cnt==1:
            return state
        mean=torch.from_numpy(self.r_mean).float().to(device)
        var=torch.from_numpy(self.r_var).float().to(device)
        print('s',state)
        print('m',mean)
        print('v',var)
        # print(mean)
        # mean=mean.repeat(batch_size,1).to(device)
        # var=var.repeat(batch_size,1).to(device)
        return (state-mean)/(var)