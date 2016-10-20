import numpy as np

class BinomialLinearPolicy(object):
    def __init__(self, w, b):
        self.w = w
        self.b = b
    def act(self, ob):
        prob = self.get_action_prob(ob)
        a = np.random.choice(2, 1, p=prob)
        return a
    def get_parameters(self):
        return self.w, self.b
    def set_parameters(self, w, b):
        self.w = w
        self.b = b
    def get_action_prob(self, ob):
        x = np.dot(self.w, ob) + self.b
        prob = self._softmax(x)
        return prob
    def _softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)