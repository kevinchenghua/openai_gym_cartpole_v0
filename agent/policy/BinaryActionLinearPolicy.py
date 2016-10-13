import numpy as np

class BinaryActionLinearPolicy(object):
    def __init__(self, w, b):
        self.w = w
        self.b = b
    def act(self, ob):
        y = np.dot(self.w, ob) + self.b
        a = int(y < 0)
        return a
    def get_parameters(self):
        return self.w, self.b
    def set_parameters(self, w, b):
        self.w = w
        self.b = b