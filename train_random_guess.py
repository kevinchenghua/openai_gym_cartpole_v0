import gym
import numpy
import cPickle as pkl

from agent.random_guess import RandomGuess

#------------------- Create environment ---------------------

env = gym.make('CartPole-v0')

# Check Observation Space
print(env.observation_space.shape)
print(env.observation_space.high)
print(env.observation_space.low)

#------------------------------------------------------------

#-------------------- Create best agent ---------------------

## Sample related parameters
N_SAMPLES = 10000       # Total number of samples
N_STEPS = 500           # Maximum step to run a trial
W_MEAN = 0.0            # The mean of normal distribution to generate w.
W_STD = 1.0             # The standard deviation of normal distribution to generate w.
B_MEAN = 0.0            # The mean of normal distribution to generate b.
B_STD = 1.0             # The standard deviation of normal distribution to generate b.

# choose a best agent from random guess algorithm
RG = RandomGuess(env, N_SAMPLES, N_STEPS, W_MEAN, W_STD, B_MEAN, B_STD)
best_agent = RG.get_best()

# save the agent
with open("result/random_guess/agent.pkl", 'wb') as f:
    pkl.dump(best_agent, f)

#------------------------------------------------------------

#------------------- Evaluate the result --------------------

## Evaluation related parameters
EPISODES = 20           # Number of testing trials
STEPS = 10000            # Maximum step for testing

average_reward = 0.0
for i in range(EPISODES):
    total_reward = 0
    ob = env.reset()
    for t in range(STEPS):
        env.render()
        action = best_agent.act(ob)
        (ob, reward, done, _info) = env.step(action)
        total_reward += reward
        if done:
            break
    average_reward += total_reward
    print str(i+1) + "th trial with reward " + str(total_reward)
env.render(close=True)
average_reward /= EPISODES
print "Average reward " + str(average_reward) + " with " + str(EPISODES) + " trials and maximum steps " + str(STEPS)
    
#------------------------------------------------------------