# openai_gym_cartpole_v0

First step into RL with OpenAI gym CartPole-v0

## Getting Started

If you'd like to try your RL algorithm, OpenAI gym is a good place to play around with. Like MNIST for supervised learning, reinforcement learning also demand a benchmark to compare each algorithm. And this is what OpenAI aimed to do. Read more: https://gym.openai.com/.

**CartPole-v0**, one of the simplest item in OpenAI gym. In this game, there is a pole on a cart. And the goal is to balance the pole on the cart and maintain the cart in the range of the window. All you can do is applying a force on the cart from right or left. To learn more about CartPole-v0, check this site: https://gym.openai.com/envs/CartPole-v0.

Here we try simplest Linear model, and use three different algorithms to choose the policy:
* Random guessing
* Hill-climbing
* Policy gradient

### Prerequisites
This code is writing in python. To use it you will need:
* Python 2.7
* [numpy](http://docs.scipy.org/doc/numpy-1.10.0/user/install.html)
* [gym](https://github.com/openai/gym)

## Usage
### Random guessing
choose the policy:
```
python train_random_guess.py
```
This will generate a agent pickle file ```./result/random_guess/agent.pkl```. To use it:
```
python evaluate_random_guess.py
```
And you'll get some testing result in ```./result/experiment-1/```. Example:

![](images/openaigym.gif)

### Hill Climbing
choose the policy:
```
python train_hill_climb.py
```
This will generatr a agent pickle file ```./result/hill_climb/agent.pkl```.  
And you'll get training process record in ```./result/hill_climb/train/```.

### Policy Gradient
choose the policy:
```
python train_policy_gradient.py
```
This will generate a agent pickle file ```./result/policy_gradient```.  
And you'll get training process record in ```./result/policy_gradient/train/```.  
Also the image of parameters trajectory in PCA trasformation ```./result/policy_gradient/training_track.png``` and ```./result/policy_gradient/training_track_reward.png```:  
![](images/training_track.png)![](images/training_track_reward.png)

We can found that Monte-Carlo policy gradient has high variance. To reduce the variance, we could substract a baseline from the value function. And this baseline is estimate from the reward of the training histroy. To train with this method:
```
python train_policy_gradient_baseline.py
```  
You'll get the result:
![](images/training_track_baseline.png)![](images/training_track_reward_baseline.png)

The variance is reduced.
