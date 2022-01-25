import gym
import tensorflow as tf

"""
attempt 1 based on https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
and https://keon.github.io/deep-q-learning/

Using a method called Q-learning

Based on a table of possible states and actions and their values, which 
might not be the best approach for this particular env, but let's see

There are only 2 possible actions at any given state
0 - push right
1 - push left

However, there are 4 state variables with continuous values, meaning that there aren't easily definable states
"""


env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        # this is where something can be done!
        # action = env.action_space.sample()
        if observation[2] > 0:
            action = 1
        else:
            action = 0
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()



#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,4],0,0.01))
Qout = tf.matmul(inputs1,W)
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)