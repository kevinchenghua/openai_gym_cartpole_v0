import gym
import numpy
import cPickle as pkl

from agent.policy_gradient import PolicyGradient

#------------------- Create environment ---------------------

env = gym.make('CartPole-v0')

# Check Observation Space
print(env.observation_space.shape)
print(env.observation_space.high)
print(env.observation_space.low)

#------------------------------------------------------------

#-------------------- Create best agent ---------------------

## Sample related parameters
N_EPISODES = 10000           # Maximum number of episodes to find the best policy
N_STEPS = 200                # Maximum step to run a trial (because monitor is used, N_STEPS is 200 at most)
L_RATE = 0.01                # Learning rate
W_INIT_MEAN = 0.0            # The mean of normal distribution to generate w.
W_INIT_STD = 1.0             # The standard deviation of normal distribution to generate w.
B_INIT_MEAN = 0.0            # The mean of normal distribution to generate b.
B_INIT_STD = 1.0             # The standard deviation of normal distribution to generate b.

# choose a best agent from random guess algorithm
env.monitor.start('result/policy_gradient/train', force=True)
PG = PolicyGradient(env, N_EPISODES, N_STEPS, L_RATE, W_INIT_MEAN, W_INIT_STD, B_INIT_MEAN, B_INIT_STD)
env.monitor.close()
agent = PG.get_agent()


# save the agent
with open("result/policy_gradient/agent.pkl", 'wb') as f:
    pkl.dump(agent, f)

#------------------------------------------------------------

#------------------- Evaluate the result --------------------

## Evaluation related parameters
EPISODES = 20           # Number of testing trials
STEPS = 1000            # Maximum step for testing

average_reward = 0.0
for i in range(EPISODES):
    total_reward = 0
    ob = env.reset()
    for t in range(STEPS):
        env.render()
        action = agent.act(ob)
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