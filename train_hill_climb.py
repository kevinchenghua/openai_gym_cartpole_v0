import gym
import numpy
import cPickle as pkl

from agent.hill_climb import HillClimb

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
N_STEPS = 200                # Maximum step to run a trial
W_INIT_MEAN = 0.0            # The mean of normal distribution to generate w.
W_INIT_STD = 1.0             # The standard deviation of normal distribution to generate w.
W_PERT_STD = 0.5             # The standard deviation of normal distribution to perturb w.
B_INIT_MEAN = 0.0            # The mean of normal distribution to generate b.
B_INIT_STD = 1.0             # The standard deviation of normal distribution to generate b.
B_PERT_STD = 0.5             # The standard deviation of normal distribution to perturb b.

# choose a best agent from random guess algorithm
env.monitor.start('result/hill_climb/train')
HC = HillClimb(env, N_EPISODES, N_STEPS, W_INIT_MEAN, W_INIT_STD, W_PERT_STD, B_INIT_MEAN, B_INIT_STD, B_PERT_STD)
env.monitor.close()
best_agent = HC.get_best()


# save the agent
with open("result/hill_climb/agent.pkl", 'wb') as f:
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