import numpy as np
from numpy_ringbuffer import RingBuffer
from utils import deque,rgb2gray


class Memory():
  def __init__(self, stack_size, memory_size, batch_size):
    self.curr_state_stack = deque(stack_size)
    self.next_state_stack = deque(stack_size)
    self.action = deque(1)
    self.reward = deque(stack_size)
    self.memory = deque(memory_size)
    self.stack_size=stack_size
    self.batch_size=batch_size
    

  def add(self, curr_frame, action, next_frame, reward):
    self.stack_frame(curr_frame, action, next_frame, reward)
    if reward > 10.0:
      for _ in range(999):
        self.memory.append((self.curr_state_stack.array, self.action.array, self.next_state_stack.array, self.reward.array))
    self.memory.append((self.curr_state_stack.array, self.action.array, self.next_state_stack.array, self.reward.array))


  def stack_frame(self, curr_frame, action, next_frame, reward):
    self.curr_state_stack.append(curr_frame)
    self.action.append(action)
    self.next_state_stack.append(next_frame)
    self.reward.append(reward)


  def samples(self):
    memo_len = self.memory._len()
    index = np.random.choice(np.arange(memo_len-1), size = self.batch_size, replace = False)
    return [self.memory.array[_i] for _i in index]

  
  def random_memory_filler(self, env, size):
    curr_frame = env.reset()
    for _ in range(size):
      env.render()
      action = np.random.choice([1,0,0,0,0,0,0,0,0],9,replace=False) # take a random action
      action_int = int(np.argmax(action))
      next_frame, reward, done, info = env.step(action_int)
      if reward == 0.0:
        reward = -0.1
      if done:
        reward = -100.0
      self.add(rgb2gray(curr_frame), action, rgb2gray(next_frame), reward)
      if done:
        env.reset()
      #print('k')
  