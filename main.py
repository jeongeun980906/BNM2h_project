from kukaCamGymEnv import KukaCamGymEnv
import time
import matplotlib.pyplot as plt
import numpy as np
from sac import Agent
def main():
    env = KukaCamGymEnv(renders=True, isDiscrete=False)
    agent=Agent()
    EPISODE=100
    res=[]
    for e in range(EPISODE):
        score=0
        state=env.reset()
        done = False
        while (not done):
            action,_ = agent.sample_action(torch.from_numpy(state).unsqueeze(0).to('cuda').float())
            action = action.detach().tolist()[0]
            next_state, reward, done, info = env.step(action)
            agent.memory.store(state,action,reward,next_state,done)
            score+=reward
            state=next_state
            env.render()
            if len(agent.memory.buffer)>200:
                print('training,,')
                for i in range(3):
                    agent.train()
                    agent.update_param()

        print('Episode: {}, Score: {}'.format(e,score))
        res.append(score)
        if e%20==0 and e != 0:
            agent.save_nn(e,score)
    plt.plot(res)
    plt.show()

if __name__ == "__main__":
  main()