import gym
import numpy as np
from memory import Memory
from DQmodel import DQNetwork
from utils import rgb2gray, take_random_action
import cv2

stack_size=4

dqn = DQNetwork()
memo = Memory(4, 10000, 1024)
env = gym.make('MsPacman-v0')
memo.random_memory_filler(env, 10000)

curr_frames = np.random.rand(1, 172, 160, 4)

env.reset()
decay = -1

for i_episode in range(300):
  observation = env.reset()
  decay = decay + 1
  total_reward = 0
  for _ in range(10000):
    env.render()
    samples = memo.samples()
  
    if(take_random_action(1.0, 0.01, 0.001, decay)):
      next_action = np.random.choice([1,0,0,0,0,0,0,0,0],9,replace=False)
      
    else:
      next_action = dqn.get_action(curr_frames)
    
    next_frames = []
    ack = np.argmax(next_action)

    for i in range(stack_size):
      env.render()
      next_frame, new_reward, done, info = env.step(ack)
      
      if new_reward == 0.0:
        new_reward = -1.0
      
      if done:
        new_reward = -100.0
      
      total_reward = total_reward + new_reward
      next_frame = rgb2gray(next_frame)
      next_frames.append(next_frame)
      memo.add(curr_frames[0,:,:,i], next_action, next_frame, new_reward)
      
      if done:
        env.reset()
        decay = decay + 1
        print('Loss: {}, Reward: {}'.format(float(loss), total_reward))
        total_reward = 0
    
    curr_frames = np.array(next_frames).reshape(1,172,160,4)

    curr_frame_stack = np.array([np.array(sample[0]).reshape(172, 160, 4) for sample in samples])
    curr_action_stack = np.array([np.array(sample[1]) for sample in samples])
    next_frame_stack = np.array([np.array(sample[2]).reshape(172, 160, 4) for sample in samples])
    reward = np.sum(np.array([np.sum(np.array(sample[3])) for sample in samples]))      
    loss = dqn.minimize_loss(curr_frame_stack, curr_action_stack, next_frame_stack, reward)