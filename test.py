import gym

"""
attempt 1 based on https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

Using a method called Q-learning

Based on a table of possible states and actions and their values, which 
might not be the best approach for this particular env, but let's see
"""


env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        # this is where something can be done!
        # action = env.action_space.sample()
        action = 1
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

