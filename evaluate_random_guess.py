import gym
import numpy
import cPickle as pkl

#-------------------- Utility functions ---------------------
def video_callable(episode_id):
    return True
#------------------------------------------------------------



#############################################################



#------------------- Create environment ---------------------

env = gym.make('CartPole-v0')

# Check Observation Space
print(env.observation_space.shape)
print(env.observation_space.high)
print(env.observation_space.low)

#------------------------------------------------------------

#------------------ Load the best agent ---------------------

with open("result/random_guess/agent.pkl", 'r') as f:
    agent = pkl.load(f)

#------------------------------------------------------------

#------------------- Evaluate the result --------------------

## Evaluation related parameters
EPISODES = 20           # Number of testing trials
STEPS = 10000            # Maximum step for testing (though with the moniter its maximum is 200)

env.monitor.start('result/random_guess/experiment-1', video_callable=video_callable)
for i in range(EPISODES):
    total_reward = 0
    ob = env.reset()
    for t in range(STEPS):
        action = agent.act(ob)
        (ob, reward, done, _info) = env.step(action)
        total_reward += reward
        if done:
            break
    print str(i+1) + "th trial with reward " + str(total_reward)

print "Average reward " + str(average_reward) + " with " + str(EPISODES) + " trials and maximum steps " + str(STEPS)

env.monitor.close()
    
#------------------------------------------------------------