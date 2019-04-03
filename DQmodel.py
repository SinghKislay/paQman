import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

class DQNetwork(Model):
  def __init__(self):
    super(DQNetwork,self).__init__()
    self.Conv1 = tf.keras.layers.Conv2D(64, (32, 32), (8, 8), padding = 'same', activation = tf.nn.relu)
    self.Conv2 = tf.keras.layers.Conv2D(128, (8, 8), (4, 4), padding = 'same', activation = tf.nn.relu)
    self.Conv3 = tf.keras.layers.Conv2D(256, (3, 3), (1, 1), padding = 'same', activation = tf.nn.relu)
    self.BatchNorm = tf.keras.layers.BatchNormalization()
    self.Flatten = tf.keras.layers.Flatten()
    self.Dence1 = tf.keras.layers.Dense(units = 256)
    self.Dence2 = tf.keras.layers.Dense(units = 128)
    self.LeakyRelu = tf.keras.layers.LeakyReLU()
    self.Dence3 = tf.keras.layers.Dense(units = 9)
    self.mse = tf.keras.losses.MeanSquaredError()
    

  def call(self, frame_stack):
    frame_stack = self.Conv1(frame_stack)
    frame_stack = self.BatchNorm(frame_stack)
    frame_stack = self.Conv2(frame_stack)
    frame_stack = self.Conv3(frame_stack)
    frame_stack = self.Flatten(frame_stack)
    frame_stack = self.Dence1(frame_stack)
    frame_stack = self.LeakyRelu(frame_stack)
    frame_stack = self.Dence2(frame_stack)
    frame_stack = self.LeakyRelu(frame_stack)
    output = self.Dence3(frame_stack)
    return output
  
  def get_action(self, curr_stack):
    curr_stack = tf.cast(curr_stack, dtype = tf.float32)
    q = self(curr_stack)
    next_action = tf.one_hot(tf.math.argmax(q, axis = 1), depth = 9, axis=-1)
    return next_action

  #@tf.function
  def minimize_loss(self, curr_stack, curr_action, next_stack, reward):
    curr_stack = tf.cast(curr_stack, dtype = tf.float32)
    curr_action = tf.cast(curr_action , dtype = tf.float32)
    next_stack = tf.cast(next_stack, dtype = tf.float32)

    with tf.GradientTape() as tape:
      q = self(curr_stack)
      q_reward = tf.math.reduce_sum(tf.math.multiply(q,curr_action))

      q_target = self(next_stack)
      next_action = tf.one_hot(tf.math.argmax(q_target, axis = 1), depth = 9, axis=-1)
      q_target_reward = tf.math.reduce_sum(tf.math.multiply(q_target, next_action))
      loss = self.mse([reward+0.9*q_target_reward],[q_reward])
    

    optimizer = tf.keras.optimizers.Adam()
    gradients = tape.gradient(loss, self.trainable_variables)
    optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    return loss
  
  




