import tensorflow as tf
from DQmodel import DQNetwork
import numpy as np
import gym
from utils import rgb2gray
import time
env = gym.make('MsPacman-v0')
img = env.reset()
img = rgb2gray(img)
dqn = DQNetwork()
dqn.load_weights('./Checkpoints/DQNetwork-Epoch20')
img_stack = np.array([img]*4).reshape(1,84,84,4)
action = dqn.get_action(img_stack)

while True:
  img_stack = []
  for _ in range(4):
    env.render()
    obs,_,_,_ = env.step(np.argmax(action))
    time.sleep(1/30)
    obs = rgb2gray(obs)
    img_stack.append(obs)
  nxt_stack = np.array(img_stack).reshape(1,84,84,4)
  action = dqn.get_action(nxt_stack)
  

  

