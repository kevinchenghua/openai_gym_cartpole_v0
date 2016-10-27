import numpy as np
import os
from sklearn.decomposition import PCA
from policy.BinomialLinearPolicy import BinomialLinearPolicy

class PolicyGradientBaseline(object):
    """This is a class of choosing policy
    
    This class use the policy gradient algorithm to update policy.
    
    Attributes:
        env: The gym environment.
        n_episodes(int): The number episodes to learn the best policy.
        n_steps(int): The number of maximum steps for a trial.
        w_init_mean(double): The mean of normal distribution to generate w.
        w_init_std(double): The standard deviation of normal distribution to generate w.
        b_init_mean(double): The mean of normal distribution to generate b.
        b_init_std(double): The standard deviation of normal distribution to generate b.
        agent(BinomialLinearPolicy): The best agent choosed.
        w_track(list): w training track.
        b_track(list): b training track.
        reward_track(list): reward training track.
    
    """
    def __init__(self, env, n_episodes=10000, n_steps=200, lr=0.01, baseline_a=0.1, w_init_mean=0.0, w_init_std=1.0, b_init_mean=0.0, b_init_std=1.0):
        self.env = env
        self.n_episodes = n_episodes
        self.n_steps = n_steps
        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std
        self.b_init_mean = b_init_mean
        self.b_init_std = b_init_std
        
        self.agent, self.w_track, self.b_track, self.reward_track = self._train(n_steps, lr, baseline_a)
        
    def _train(self, n_steps, lr, baseline_a):
        """This is a helper method for __init__.
        
        This method try to update the polocy with policy gradient.
        
        Args:
            n_steps(int): The maximum steps of a trial.
            lr(double): The learning rate.
            baseline_a(double): The contribution for the current reward to the baseline.
        Returns:
            agent(BinomialLinearPolicy): The agent after updated.
            w_list(list): List of w for each update.
            b_list(list): List of b for each update.
        """
        agent = self._generate_agent(self.w_init_mean, self.w_init_std, self.b_init_mean, self.b_init_std)
        w_list = []
        b_list = []
        reward_list = []
        baseline = 0.0
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
            baseline = (1.0-baseline_a) * baseline + baseline_a * total_reward
                    
            print "with reward "+str(total_reward)
            #if total_reward == n_steps:
            #    print "finish the update."
            #    break
                
            w, b = agent.get_parameters()
            w_list.append(w)
            b_list.append(b)
            reward_list.append(total_reward)
            w_grad = 0.0
            b_grad = 0.0
            for t in range(len(state_list)):
                wg, bg = agent.get_grad(state_list[t], action_list[t])
                value = reward_array[t:].sum() - (baseline - t)
                w_grad += wg * value
                b_grad += bg * value
            
            agent.set_parameters(w+w_grad*lr, b+b_grad*lr)
            
        print "Find agent with reward: " + str(total_reward) + " in " + str(i+1) + " epsoides."
        return agent, w_list, b_list, reward_list
            
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
    
    def get_pca_track(self):
        """Method to get pca of parameters track
        
        Returns:
            (numpy.ndarray): array of [[pca1, pca2, reward]] with shape (n_episodes, 3).
        
        """
        w_track = np.stack(self.w_track).reshape((len(self.w_track), -1))
        b_track = np.stack(self.b_track)
        param_track = np.hstack((w_track, b_track))
        pca = PCA(n_components=2)
        pca.fit(param_track)
        print "The variance ratio explained by each component: " + str(pca.explained_variance_ratio_)
        return np.hstack((pca.transform(param_track), np.array(self.reward_track)[:,None]))
    
    def get_agent(self):
        return self.agent