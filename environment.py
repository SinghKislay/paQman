import gym
import numpy as np
from memory import Memory
from DQmodel import DQNetwork
from utils import rgb2gray, take_random_action, get_data
import cv2
import tensorflow as tf

l=0
dqn = DQNetwork()
dqn.load_weights('./Checkpoints/DQNetwork-Epoch20')
memo = Memory(4, 1000000, 32)
env = gym.make('MsPacman-v0')

memo.random_memory_filler(env, 32)
curr_frames = np.random.rand(1, 84, 84, 4)
writer = tf.summary.create_file_writer('./Performance_summary')
env.reset()
decay = -1
stack_size=4
for i_episode in range(200):
  decay = decay + 1
  total_reward = 0
  for _ in range(10000):
    env.render()
    samples = memo.samples()
  
    
    next_action = dqn.get_action(curr_frames)
    
    next_frames = []
    ack = np.argmax(next_action)

    for i in range(stack_size):
      env.render()
      next_frame, new_reward, done, info = env.step(ack)
      
      if new_reward == 0.0:
        new_reward = -0.1

      if done:
        l=l+1
        new_reward = -100.0
        env.reset()
        decay = decay + 1
        print('Loss: {}, Reward: {}'.format(float(loss), total_reward))
        with writer.as_default():
          a = tf.summary.scalar('loss', loss, step = l)
          b = tf.summary.scalar('reward', total_reward, step = l)
        total_reward = 0
      
      total_reward = total_reward + new_reward
      next_frame = rgb2gray(next_frame)
      next_frames.append(next_frame)
      memo.add(curr_frames[0,:,:,i], next_action, next_frame, new_reward)
 
    curr_frames = np.array(next_frames).reshape(1,84,84,4)

    curr_frame_stack, curr_action_stack, next_frame_stack, reward = get_data(samples)
    loss = dqn.minimize_loss(curr_frame_stack, curr_action_stack, next_frame_stack, reward)

  if (i_episode % 10 == 0):
    m_name = './Checkpoints/DQNetwork-Epoch'+str(i_episode+1)
    dqn.save_weights(m_name)