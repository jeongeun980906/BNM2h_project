from kukaCamGymEnv import KukaCamGymEnv
import time
import matplotlib.pyplot as plt

def main():

  environment = KukaCamGymEnv(renders=True, isDiscrete=False)

  motorsIds = []
  #motorsIds.append(environment._p.addUserDebugParameter("posX",0.4,0.75,0.537))
  #motorsIds.append(environment._p.addUserDebugParameter("posY",-.22,.3,0.0))
  #motorsIds.append(environment._p.addUserDebugParameter("posZ",0.1,1,0.2))
  #motorsIds.append(environment._p.addUserDebugParameter("yaw",-3.14,3.14,0))
  #motorsIds.append(environment._p.addUserDebugParameter("fingerAngle",0,0.3,.3))

  dv = 1
  motorsIds.append(environment._p.addUserDebugParameter("posX", -dv, dv, 0))
  motorsIds.append(environment._p.addUserDebugParameter("posY", -dv, dv, 0))
  motorsIds.append(environment._p.addUserDebugParameter("posZ", -dv, dv, 0))
  motorsIds.append(environment._p.addUserDebugParameter("yaw", -dv, dv, 0))
  motorsIds.append(environment._p.addUserDebugParameter("fingerAngle", 0, 0.3, .3))

  done = False
  while (not done):

    action = []
    for motorId in motorsIds:
      action.append(environment._p.readUserDebugParameter(motorId))
    print(action)
    state, reward, done, info = environment.step(action)
    #state = rgbd
    obs = environment.getExtendedObservation()


if __name__ == "__main__":
  main()
