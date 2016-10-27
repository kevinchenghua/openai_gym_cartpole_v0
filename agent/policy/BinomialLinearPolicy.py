import numpy as np
import theano
import theano.tensor as T

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]

class BinomialLinearPolicy(object):
    def __init__(self, w, b):
        self.w = w
        self.b = b
    def act(self, ob):
        prob = self.get_action_prob(ob)
        a = np.random.choice(2, 1, p=prob)
        return a[0]
    def get_parameters(self):
        return self.w, self.b
    def set_parameters(self, w, b):
        self.w = w
        self.b = b
    def get_action_prob(self, ob):
        x = np.dot(self.w, ob) + self.b
        prob = self._softmax(x)
        return prob
    def get_grad(self, ob, a):
        b_grad = -self.get_action_prob(ob)
        b_grad[a] = 1 + b_grad[a]
        w_grad = np.dot(b_grad[np.newaxis].T, ob[np.newaxis])
        return w_grad, b_grad
    def _softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
        
class BinomialLinearPolicyTheano(object):
    def __init__(self, env, w=None, b=None, w_mean=0.0, w_std=1.0, b_mean=0.0, b_std=1.0):
        self.params = self._init_params(w, b, w_mean, w_std, b_mean, b_std)
        self.inputs, self.outputs = self._build_forward()
        self.grads = self._build_backward()
        self.f_act = self._build_function()
        
        
    def _init_params(self, w, b , w_mean, w_std, b_mean, b_std):
        params = {}
        shared_params = {}
        
        # create the numpy parameters
        params['w'] = w
        params['b'] = b
        if w==None:
            params['w'] = np.random.normal(w_mean, w_std, self.env.observation_space.shape+(self.env.action_space.n,))
        if b==None:    
            params['b'] = np.random.normal(b_mean, b_std, self.env.action_space.n)
        
        # convert the numpy weights to theano shared variable
        for key, value in params.iteritems():
            shared_params[key] = theano.shared(value, name=key)
            
        return shared_params
        
    def _build_forward(self):
        # input variable
        states = T.matrix()     # step x observation_dimension
        actions = T.imatrix()   # step x action_dimention
        advantages = T.vetcor() # step
        # computation graph
        linear = T.dot(states, self.params['w']) + self.params['b']
        probs = T.nnet.softmax(linear)
        action_probs = (actions * probs).sum(1)
        loss = -T.log(action_probs * advantages).sum()
        
        inputs = [states, actions, advantages]
        outputs = [probs, loss]
        
        return inputs, outputs
        
    def _build_function(self):
        states, actions, _ = self.inputs
        probs, _ = self.outputs
        rng = RandomStreams(1234)
        act = rng.multinomial(pvals=probs)
        f_act = theano.function([states],act)
        
        return f_act
    
    def _build_backward(self):
        _, loss = self.outputs
        grads = T.grad(loss, wrt=itemlist(self.params))
        
        return grads
    
    def act(self, state):
        return self.f_act(state)
        
    def get_params(self):
        return self.params