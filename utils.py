import numpy as np
import cv2

def rgb2gray(rgb):
  img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).reshape(210,160)[:172][:]/255
  img = res = cv2.resize(img,(84, 84), interpolation = cv2.INTER_CUBIC)
  return np.array(img)

def take_random_action(explore_start, explore_stop, decay_rate, decay_step):
  exp_exp_tradeoff = np.random.rand()
  explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
  if (explore_probability > exp_exp_tradeoff):
    return True
  else:
    return False


class deque():
  def __init__(self, max_len):
    self.array = [] 
    self.max_len = max_len
  def append(self, data):
    if len(self.array) == self.max_len:
      self.array.pop(0)
      self.array.append(data)
    else:
      self.array.append(data)
  def _len(self):
    return len(self.array)

def get_data(samples):
  cfs = []
  cas = []
  nfs = []
  r = []
  for sample in samples:
    cfs.append(np.array(sample[0]).reshape(84, 84, 4))
    cas.append(np.array(sample[1]))
    nfs.append(np.array(sample[2]).reshape(84, 84, 4))
    r.append(np.sum(np.array(sample[3])))

  return np.array(cfs), np.array(cas), np.array(nfs), np.sum(r)



    