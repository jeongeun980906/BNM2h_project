import gym
import time
import matplotlib.pyplot as plt
import numpy as np
from sac import Agent
import torch
import pybulletgym

def main():
    env = gym.make('AntPyBulletEnv-v0')
    print(env.action_space)
    print(env.observation_space)
    agent=Agent(action_size=8,action_scale=1,state_size=28)
    EPISODE=100
    res=[]
    flag=False
    for e in range(EPISODE):
        env.render()
        score=0
        step=0
        state=env.reset()
        done = False
        while (not done):
            # agent.moving_avarage.run(state)
            # if flag:
            #     nstate=(state-agent.moving_avarage.r_mean)/agent.moving_avarage.r_var
            #     nstate=np.nan_to_num(nstate)
            #     #print(nstate)
            #     action,_ = agent.sample_action(torch.from_numpy(nstate).unsqueeze(0).to('cuda').float())
            # else:
            #     action,_ = agent.sample_action(torch.from_numpy(state).unsqueeze(0).to('cuda').float())
            action,_ = agent.sample_action(torch.from_numpy(state).unsqueeze(0).to('cuda').float())
            action = action.detach().tolist()[0]
            #print(action)
            next_state, reward, done, info = env.step(action)
            agent.memory.store(state,action,reward,next_state,done)
            score+=reward
            state=next_state
            env.render()
            if len(agent.memory.buffer)>400:
                #flag=True
                #print('training')
                for i in range(3):
                    agent.train()
                    agent.update_param()
            # if step>100:
            #     break
            step+=1

        print('Episode: {}, Score: {}'.format(e,score))
        res.append(score)
        if e%200==0 and e != 0:
            agent.save_nn(e,score)
    plt.plot(res)
    plt.show()

if __name__ == "__main__":
  main()