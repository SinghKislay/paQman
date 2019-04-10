import gym
import numpy as np
import time
import cv2
from utils import rgb2gray

env = gym.make('MsPacman-v0')
ob=env.reset()
print(ob.shape)

ob = rgb2gray(ob)

print(ob)
cv2.imshow('img',ob)
cv2.waitKey()
for _ in range(4):
    env.render()
    #print(env.action_space)
    action = env.action_space.sample() # take a random action
    observation, reward, done, _ = env.step(action)
    #print(action)
    #print(reward)
    #print(observation.shape)
    time.sleep(4)
env.close()