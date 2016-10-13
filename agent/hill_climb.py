import numpy as np
import os
from policy.BinaryActionLinearPolicy import BinaryActionLinearPolicy

class HillClimb(object):
    """This is a class of choosing policy
    
    This class generate random BinaryActionLinearPolicy agents and choose the best one.
    
    Attributes:
        env: The gym environment.
        n_episodes(int): The number episodes to learn the best policy.
        n_steps(int): The number of maximum steps for a trial.
        w_init_mean(double): The mean of normal distribution to generate w.
        w_init_std(double): The standard deviation of normal distribution to generate w.
        w_pert_std(double): The standard deviation of normal distribution to perturb w of the policy.
        b_init_mean(double): The mean of normal distribution to generate b.
        b_init_std(double): The standard deviation of normal distribution to generate b.
        b_pert_std(double): The standard deviation of normal distribution to perturb b of the policy.
        best_agent(BinaryActionLinearPolicy): The best agent choosed.
    
    """
    def __init__(self, env, n_episodes=10000, n_steps=250, w_init_mean=0.0, w_init_std=1.0, w_pert_std=0.5, b_init_mean=0.0, b_init_std=1.0, b_pert_std=0.5):
        self.env = env
        self.n_episodes = n_episodes
        self.n_steps = n_steps
        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std
        self.w_pert_std = w_pert_std
        self.b_init_mean = b_init_mean
        self.b_init_std = b_init_std
        self.b_pert_std = b_pert_std
        
        self.best_agent = self._train(n_steps)
        
    def _train(self, n_steps=200):
        """This is a helper method for __init__.
        
        This method try n_samples times and return the best parameter.
        
        Args:
            n_steps(int): The maximum steps for a trial.
        Returns:
            best_agent(BinaryActionLinearPolicy): The best agent with the highest reward.
        """
        best_reward = 0
        best_agent = self._generate_agent(self.w_init_mean, self.w_init_std, self.b_init_mean, self.b_init_std)
        for i in range(self.n_episodes):
            print str(i+1)+"th trial ",
            total_reward = 0
            w, b = best_agent.get_parameters()
            agent = self._generate_agent(w, self.w_pert_std, b, self.b_pert_std)
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
            if best_reward == n_steps:
                break
        print "Find best agent with reward: " + str(best_reward) + " in " + str(i+1) + " epsoides."
        return best_agent
            
    def _generate_agent(self, w_mean, w_std, b_mean, b_std):
        """Helper method for _train.
        This is a method to generate a BinaryActionLinearPolicy with normal distributed parameters.
        
        Args:
            w_mean(double): The mean of normal distribution to generate w.
            w_std(double): The standard deviation of normal distribution to generate w.
            b_mean(double): The mean of normal distribution to generate b.
            b_std(double): The standard deviation of normal distribution to generate b.
        
        Retruns:
            (BinaryActionLinearPolicy)
        """
        w = np.random.normal(w_mean, w_std, self.env.observation_space.shape)
        b = np.random.normal(b_mean, b_std)
        return BinaryActionLinearPolicy(w,b)
        
    def get_best(self):
        """Method for get best agent."""
        return self.best_agent