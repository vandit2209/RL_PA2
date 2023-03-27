
import numpy as np
import random
import torch
import torch.nn as nn  
import torch.nn.functional as F
from collections import namedtuple, deque
import torch.optim as optim
import datetime
import gym
from gym.wrappers.record_video import RecordVideo
import glob
import io
import base64
import matplotlib.pyplot as plt
from IPython.display import HTML
from pyvirtualdisplay import Display
import tensorflow as tf
from IPython import display as ipythondisplay
from PIL import Image
import tensorflow_probability as tfp
import random
import torch
import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn  
import torch.nn.functional as F
import os
import wandb
import shutup
shutup.please()

'''
Please refer to the first tutorial for more details on the specifics of environments
We've only added important commands you might find useful for experiments.
'''

'''
List of example environments
(Source - https://gym.openai.com/envs/#classic_control)

'Acrobot-v1'
'Cartpole-v1'
'MountainCar-v0'
'''



# print(state_shape)
# print(no_of_actions)
# print(env.action_space.sample())
# print("----")

'''
# Understanding State, Action, Reward Dynamics

The agent decides an action to take depending on the state.

The Environment keeps a variable specifically for the current state.
- Everytime an action is passed to the environment, it calculates the new state and updates the current state variable.
- It returns the new current state and reward for the agent to take the next action

'''

# state = env.reset()   
''' This returns the initial state (when environment is reset) '''

# print(state)
# print("----")

# action = env.action_space.sample()  
''' We take a random action now '''

# print(action)
# print("----")

# next_state, reward, done, info = env.step(action) 
''' env.step is used to calculate new state and obtain reward based on old state and action taken  ''' 

# print(next_state)
# print(reward)
# print(done)
# print(info)
# print("----")

# %%
'''
### Q Network & Some 'hyperparameters'
QNetwork1:
Input Layer - 4 nodes (State Shape) \
Hidden Layer 1 - 64 nodes \
Hidden Layer 2 - 64 nodes \
Output Layer - 2 nodes (Action Space) \
Optimizer - zero_grad()

QNetwork2: Feel free to experiment more
'''


'''
Bunch of Hyper parameters (Which you might have to tune later **wink wink**)
'''



class QNetwork1(nn.Module):

    def __init__(self, state_size, action_size, seed, hidden_size, hidden_layers):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork1, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layers = nn.ModuleList()
        self.hidden_layers = hidden_layers
        self.layers.append(nn.Linear(state_size, hidden_size)) # input layer
        for i in range(self.hidden_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        self.layers.append(nn.Linear(hidden_size, action_size)) # output layer

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.layers[0](state)) # input activation
        for i in range(self.hidden_layers):
            x = F.relu(self.layers[i+1](x))
        return self.layers[-1](x)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
    
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class EGreedyTutorialAgent():

    def __init__(self, state_size, action_size, seed, hidden_size, hidden_layers, buffer_size, batch_size, gamma, learning_rate, update_frequency):
        self.batch_size = batch_size
        ''' Agent Environment Interaction '''
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        ''' Q-Network '''
        self.qnetwork_local = QNetwork1(state_size, action_size, seed, hidden_size=hidden_size, hidden_layers = hidden_layers).to(device)
        self.qnetwork_target = QNetwork1(state_size, action_size, seed, hidden_size=hidden_size, hidden_layers = hidden_layers).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        ''' Replay memory '''
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)

        ''' Initialize time step (for updating every UPDATE_EVERY steps)           -Needed for Q Targets '''
        self.t_step = 0
        self.gamma = gamma
        self.update_frequency = update_frequency

    def step(self, state, action, reward, next_state, done):

        ''' Save experience in replay memory '''
        self.memory.add(state, action, reward, next_state, done)
        
        ''' If enough samples are available in memory, get random subset and learn '''
        if len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        """ +Q TARGETS PRESENT """
        ''' Updating the Network every 'UPDATE_EVERY' steps taken '''      
        self.t_step = (self.t_step + 1) % self.update_frequency
        if self.t_step == 0:

            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def act(self, state, eps=0.):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        ''' Epsilon-greedy action selection (Already Present) '''
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """ +E EXPERIENCE REPLAY PRESENT """
        states, actions, rewards, next_states, dones = experiences

        ''' Get max predicted Q values (for next states) from target model'''
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        ''' Compute Q targets for current states '''
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        ''' Get expected Q values from local model '''
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        ''' Compute loss '''
        loss = F.mse_loss(Q_expected, Q_targets)

        ''' Minimize the loss '''
        self.optimizer.zero_grad()
        loss.backward()
        
        ''' Gradiant Clipping '''
        """ +T TRUNCATION PRESENT """
        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()


class SoftmaxTutorialAgent():

    def __init__(self, state_size, action_size, seed, hidden_size, hidden_layers, buffer_size, batch_size, gamma, learning_rate, update_frequency):
        self.batch_size = batch_size
        ''' Agent Environment Interaction '''
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        ''' Q-Network '''
        self.qnetwork_local = QNetwork1(state_size, action_size, seed, hidden_size=hidden_size, hidden_layers = hidden_layers).to(device)
        self.qnetwork_target = QNetwork1(state_size, action_size, seed, hidden_size=hidden_size, hidden_layers = hidden_layers).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        ''' Replay memory '''
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)

        ''' Initialize time step (for updating every UPDATE_EVERY steps)           -Needed for Q Targets '''
        self.t_step = 0
        self.gamma = gamma
        self.update_frequency = update_frequency
    
    def step(self, state, action, reward, next_state, done):

        ''' Save experience in replay memory '''
        self.memory.add(state, action, reward, next_state, done)
        
        ''' If enough samples are available in memory, get random subset and learn '''
        if len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        """ +Q TARGETS PRESENT """
        ''' Updating the Network every 'UPDATE_EVERY' steps taken '''      
        self.t_step = (self.t_step + 1) % self.update_frequency
        if self.t_step == 0:

            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def act(self, state, tau=0.):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        q_values = action_values.cpu().data.numpy()[0]
        _max = np.max(q_values)
        numerator = np.exp((q_values - _max)/tau)
        denominator = np.sum(numerator)
        prob = numerator / denominator
        arm_id_selected = np.random.choice(np.arange(self.action_size), p = prob)
        ''' Epsilon-greedy action selection (Already Present) '''
        return arm_id_selected

    def learn(self, experiences):
        """ +E EXPERIENCE REPLAY PRESENT """
        states, actions, rewards, next_states, dones = experiences

        ''' Get max predicted Q values (for next states) from target model'''
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        ''' Compute Q targets for current states '''
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        ''' Get expected Q values from local model '''
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        ''' Compute loss '''
        loss = F.mse_loss(Q_expected, Q_targets)

        ''' Minimize the loss '''
        self.optimizer.zero_grad()
        loss.backward()
        
        ''' Gradiant Clipping '''
        """ +T TRUNCATION PRESENT """
        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()

def dqn(env = None, n_episodes=10000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, agent = None):

    scores = []                 
    ''' list containing scores from each episode '''
    scores_count = 0

    scores_window_printing = deque(maxlen=10) 
    ''' For printing in the graph '''
    
    scores_window= deque(maxlen=100)  
    ''' last 100 scores for checking if the avg is more than 195 '''

    eps = eps_start                    
    ''' initialize epsilon '''
    terminate = 0

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 

        scores_window.append(score)       
        scores_window_printing.append(score)   
        ''' save most recent score '''           

        eps = max(eps_end, eps_decay*eps) 
        ''' decrease epsilon '''

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")  
        if i_episode % 10 == 0: 
            scores.append(np.mean(scores_window_printing))
            scores_count += 1        
        if i_episode % 100 == 0: 
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=195.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            terminate = i_episode
            break
    return [np.array(scores),i_episode-100, scores_count, i_episode]
def run10times(_type = "egreedy"):
    config_defaults = {
    "model": "CartPole-v1",
    "buffer":int(1e5),
    "learning_rate":5e-4,
    "clip": 1.0,
    "update_frequency": 20,
    "batch_size":64,
    "gamma":0.99,
    "hidden_size": 128,
    "hidden_layers": 2
    }

    wandb.init(config=config_defaults)
    config = wandb.config
    print("MODEL", config.model)
    env = gym.make(config.model)
    env.seed(0)

    state_shape = env.observation_space.shape[0]
    # no_of_actions = env.action_space.n
    action_shape = env.action_space.n
    state = env.reset() 
    action = env.action_space.sample() 
    next_state, reward, done, info = env.step(action)
    temp = []
    max_count = 0
    if _type == "egreedy":
        for run in range(10):
            print(f"\r{run}", end = "")
            # begin_time = datetime.datetime.now()
            agent = EGreedyTutorialAgent(state_size=state_shape,action_size = action_shape,seed = 0, hidden_size= config.hidden_size, hidden_layers=config.hidden_layers,buffer_size= config.buffer, batch_size= config.batch_size, learning_rate=config.learning_rate, update_frequency=config.update_frequency, gamma = config.gamma)
            result = dqn(env = env, agent=agent)
            print("DQN")
            # time_taken = datetime.datetime.now() - begin_time
            temp.append(result)
            max_count = max(max_count, result[2])
    elif _type == "softmax":
        for run in range(10):
            print(f"\r{run}", end = "")
            # begin_time = datetime.datetime.now()
            agent = SoftmaxTutorialAgent(state_size=state_shape,action_size = action_shape,seed = 0, hidden_size=config.hidden_size, hidden_layers= config.hidden_layers, buffer_size= config.buffer, batch_size= config.batch_size, learning_rate=config.learning_rate, update_frequency=config.update_frequency, gamma=config.gamma)
            result = dqn(env = env, agent=agent)
            print("DQN")
            # time_taken = datetime.datetime.now() - begin_time
            temp.append(result)
    final_result = []
    for elm in temp:
        """ 
        since the experiment will terminate in random amount of steps
        the graph must contain the maximum_amount of steps we are plotting through
        assuming the ones who terminated before max_t will contribute constant 195 to the average
        """
        final_result.append(elm[0] + [195.0]*(max_count - elm[2]))
        
    final_result = np.array(final_result)
    print("ARRY",final_result.shape)
    final_result = np.average(final_result, axis = 0)
    ep = [i for i in range(len(final_result))]
    name = f"{_type}_bu_{config.buffer}_lr_{config.learning_rate}_hz_{config.hidden_size}_hl_{config.hidden_layer}_c_{config.clip}_uf_{config.update_frequency}_bs_{config.batch_size}_g_{config.gamma}"
    wandb.init(name=name)
    plt.title(f'DQN on {config.model} (using {_type}) for every 10 episodes \nAvergaed over 10 runs')
    plt.xlabel('Episodes')
    plt.ylabel('Average Rewards')
    plt.plot(ep, result)
    wandb.log({"chart": plt})
    path = os.getcwd()
    plt.savefig(os.path.join(path, config.model, name),".png")
# run10times(path)
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'episodes', 'goal': 'minimize'},
    'parameters':{
    "model": {"value": "CartPole-v1"},
    "buffer":{'values': [int(1e3), int(1e5), int(1e6)]},
    "learning_rate":{"values":[5e-4,1e-5,8e-6]},
    "clip": {"values":[0.5, 1]},
    "update_frequency": {"values":[15, 20, 30]},
    "batch_size":{"values":[32,64]},
    "gamma":{'values': [0.99, 0.999]},
    "hidden_size": {"values": [64, 128]},
    "hidden_layers": {"values": [2,3]}
    }
    }

sweep_id = wandb.sweep(sweep_config,project="RL_PA2")
wandb.agent(sweep_id,project="RL_PA2",function=run10times,count=50)
