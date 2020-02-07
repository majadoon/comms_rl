import gym
import numpy as np

env = gym.make('CartPole-v0')

done = False
# to count the cart moves
count = 0

#reset the environment before actually using it
observation = env.reset()
print('observation = ', observation)

while not done:
    env.render()   #to see the environment in action

    count += 1

    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)

    if done:
        break

#as the actions are random, it falls quickly after some movements
print('game lasted ', count, 'moves')
