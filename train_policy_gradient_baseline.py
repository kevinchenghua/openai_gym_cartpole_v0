import gym
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pkl

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from agent.policy_gradient_baseline import PolicyGradientBaseline

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
L_RATE = 0.00005             # Learning rate
BASELINE_A = 0.1             # baseline_a(double): The contribution for the current reward to the baseline.
W_INIT_MEAN = 0.0            # The mean of normal distribution to generate w.
W_INIT_STD = 1.0             # The standard deviation of normal distribution to generate w.
B_INIT_MEAN = 0.0            # The mean of normal distribution to generate b.
B_INIT_STD = 1.0             # The standard deviation of normal distribution to generate b.

# choose a best agent from random guess algorithm
env.monitor.start('result/policy_gradient/baseline/train', force=True)
PG = PolicyGradientBaseline(env, N_EPISODES, N_STEPS, L_RATE, BASELINE_A, W_INIT_MEAN, W_INIT_STD, B_INIT_MEAN, B_INIT_STD)
env.monitor.close()
agent = PG.get_agent()
track = PG.get_pca_track()

# plot the training track
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('PCA component 1')
ax.set_ylabel('PCA component 2')
ax.set_zlabel('Reward')
n_line = 1000
step = track.shape[0]/n_line
for i in xrange(0, track.shape[0]-step, step):
    ax.plot(track[i:i+step+1,0], track[i:i+step+1,1], track[i:i+step+1,2], color=plt.cm.plasma(255*i/track.shape[0]))
m = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
m.set_array([0,track.shape[0]])
cbar = fig.colorbar(m)
cbar.set_label('episode')  
plt.show()


# save the agent
with open("result/policy_gradient/agent.pkl", 'wb') as f:
    pkl.dump(agent, f)
# save the plot
fig.savefig('result/policy_gradient/baseline/training_track_reward.png') 
ax.elev = 90
ax.azim = 90
fig.savefig('result/policy_gradient/baseline/training_track.png') 

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