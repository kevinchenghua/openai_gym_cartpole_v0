import numpy as np
import os
from policy.BinomialLinearPolicy import BinomialLinearPolicy

class PolicyGradient(object):
    """This is a class of choosing policy
    
    This class use the policy gradient algorithm to update policy.
    
    Attributes:
        env: The gym environment.
        n_episodes(int): The number episodes to learn the best policy.
        n_steps(int): The number of maximum steps for a trial.
        lr(double): The learning rate.
        w_init_mean(double): The mean of normal distribution to generate w.
        w_init_std(double): The standard deviation of normal distribution to generate w.
        b_init_mean(double): The mean of normal distribution to generate b.
        b_init_std(double): The standard deviation of normal distribution to generate b.
        agent(BinomialLinearPolicy): The best agent choosed.
    
    """
    def __init__(self, env, n_episodes=10000, n_steps=200, lr=0.01, w_init_mean=0.0, w_init_std=1.0, b_init_mean=0.0, b_init_std=1.0):
        self.env = env
        self.n_episodes = n_episodes
        self.n_steps = n_steps
        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std
        self.b_init_mean = b_init_mean
        self.b_init_std = b_init_std
        
        self.agent = self._train(n_steps, lr)
        
    def _train(self, n_steps, lr):
        """This is a helper method for __init__.
        
        This method try to update the polocy with policy gradient.
        
        Args:
            n_steps(int): The maximum steps of a trial.
            lr(double): The learning rate.
        Returns:
            agent(BinomialLinearPolicy): The agent after updated.
        """
        agent = self._generate_agent(self.w_init_mean, self.w_init_std, self.b_init_mean, self.b_init_std)
        for i in range(self.n_episodes):
            print str(i+1)+"th trial ",
            total_reward = 0
            state_list = []
            action_list = []
            reward_array = np.array([])
            
            ob = self.env.reset()
            for t in range(n_steps):
                action = agent.act(ob)
                state_list.append(ob)
                action_list.append(action)
                (ob, reward, done, _info) = self.env.step(action)
                reward_array = np.append(reward_array, reward)
                total_reward += reward
                if done:
                    break
                    
            print "with reward "+str(total_reward)
            if total_reward == n_steps:
                print "finish the update."
                break
                
            w, b = agent.get_parameters()
            w_grad = 0.0
            b_grad = 0.0
            for t in range(len(state_list)):
                wg, bg = agent.get_grad(state_list[t], action_list[t])
                value = reward_array[t:].sum()
                w_grad += wg * value
                b_grad += bg * value
            
            agent.set_parameters(w+w_grad*lr, b+b_grad*lr)
            
        print "Find agent with reward: " + str(total_reward) + " in " + str(i+1) + " epsoides."
        return agent
            
    def _generate_agent(self, w_mean, w_std, b_mean, b_std):
        """Helper method for _train.
        This is a method to generate a BinomailLinearPolicy with normal distributed parameters.
        
        Args:
            w_mean(double): The mean of normal distribution to generate w.
            w_std(double): The standard deviation of normal distribution to generate w.
            b_mean(double): The mean of normal distribution to generate b.
            b_std(double): The standard deviation of normal distribution to generate b.
        
        Retruns:
            (BinomialLinearPolicy)
        """
        w = np.random.normal(w_mean, w_std, (self.env.action_space.n,)+self.env.observation_space.shape)
        b = np.random.normal(b_mean, b_std, self.env.action_space.n)
        return BinomialLinearPolicy(w,b)
    
    def get_agent(self):
        return self.agent