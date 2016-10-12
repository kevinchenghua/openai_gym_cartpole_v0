import numpy as np
import os
from policy.BinaryActionLinearPolicy import BinaryActionLinearPolicy

class RandomGuess(object):
    """This is a class of choosing policy
    
    This class generate random BinaryActionLinearPolicy agents and choose the best one.
    
    Attributes:
        env: The gym environment.
        n_sample(int): The number of agent samples.
        w_mean(double): The mean of normal distribution to generate w.
        w_std(double): The standard deviation of normal distribution to generate w.
        b_mean(double): The mean of normal distribution to generate b.
        b_std(double): The standard deviation of normal distribution to generate b.
        best_agent(BinaryActionLinearPolicy): The best agent choosed.
    
    """
    def __init__(self, env, n_sample=10000, w_mean=0.0, w_std=1.0, b_mean=0.0, b_std=1.0):
        self.env = env
        self.n_sample = n_sample
        self.w_mean = w_mean
        self.w_std = w_std
        self.b_mean = b_mean
        self.b_std = b_std
        
        self.best_agent = self._evaluate_best()
        
    def _evaluate_best(self, n_steps=200):
        """This is a helper method for __init__.
        
        This method try n_sample times and return the best parameter.
        
        Args:
            n_steps(int): The maximum steps for a trial.
        Returns:
            best_agent(BinaryActionLinearPolicy): The best agent with the highest reward.
        """
        best_reward = 0
        best_agent = None
        for i in range(self.n_sample):
            print str(i+1)+"th trial ",
            total_reward = 0
            agent = self._generate_agent()
            ob = self.env.reset()
            for t in range(n_steps):
                action = agent.act(ob)
                (ob, reward, done, _info) = self.env.step(action)
                total_reward += reward
                if done:
                    break
            print "with reward "+str(total_reward)
            if total_reward > best_reward:
                print "find a better agent."
                best_agent = agent
                best_reward = total_reward
        return best_agent
            
    def _generate_agent(self):
        """Helper method for _evaluate_best.
        This is a method to generate a BinaryActionLinearPolicy with normal distributed parameters.
        """
        w = np.random.normal(self.w_mean, self.w_std, self.env.observation_space.shape)
        b = np.random.normal(self.b_mean, self.b_std)
        return BinaryActionLinearPolicy(w,b)
        
    def get_best(self):
        """Method for get best agent."""
        return self.best_agent