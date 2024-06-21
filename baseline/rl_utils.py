from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import os
from torch.utils.tensorboard import SummaryWriter

class Color:
    INFO = '\033[1;34m{}\033[0m'
    WARNING = '\033[1;33m{}\033[0m'
    ERROR = '\033[1;31m{}\033[0m'

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env, agent, n_round, max_steps, replay_buffer, minimal_size, batch_size, 
                           models, use_wandb=False, print_every=50, save_every=50, model_dir=None, train=False, log_dir=None):
    
    if train:
        writer = SummaryWriter(log_dir=log_dir)
    return_list = []
    n_group = 2
    num_pred = 0
    num_prey = 0
    for _a in env.agents:
        if _a.adversary: num_pred += 1
        else: num_prey += 1
    max_nums = [num_pred, num_prey] 
    episode_return = [[] for _ in range(n_group)]
    # for i in range(10): # ITERATION
    #     with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
    #         for i_episode in range(int(num_episodes/10)):   #
    
    if train and not os.path.exists(model_dir + '-predator'):
        os.makedirs(model_dir + '-predator')
    dones = []
    step_cts = []
    for i_round in range(n_round):
        print("\n\n[*] ROUND #{0}, NUMBER: {1}".format(i_round, max_nums))
        
        mean_rewards = [[] for _ in range(n_group)]
        state = env.reset()
        
        done = False
        acts = [np.zeros((max_nums[i],), dtype=np.int32) for i in range(n_group)]
        step_ct = 0
        while not done and step_ct < max_steps: # STEPS
            for i in range(n_group):
                if i==0:
                    # action = agent.take_action(state)
                    acts[i] = agent.take_action(state[:num_pred], train=train)
                elif i==1:
                    # acts[i] = np.random.normal([0,0],size=(num_prey,2))
                    acts[i] = models[i].act(state[num_pred:])
                    
            stack_act = np.concatenate(acts, axis=0)
            next_state, all_rewards, all_done, _ = env.step(stack_act)
            _rewards = [all_rewards[:num_pred], all_rewards[num_pred:]]
            done = all(all_done[num_pred:])
            # predator
            replay_buffer.add(state[:num_pred], acts[0], all_rewards[:num_pred], next_state[:num_pred], all_done[:num_pred])
            state = next_state
            for i in range(n_group):
                mean_rewards[i].append(sum(_rewards[i]) / max_nums[i])
                
            if train and replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                agent.update(transition_dict)
            step_ct += 1
            if done: print("> step #{}, done!!!".format(step_ct))
            if step_ct % print_every == 0:
                print("> step #{}, info: {}".format(step_ct, None))
        
        if train:
            writer.add_scalar('kill', done, i_round)
            writer.add_scalar('step_ct', step_ct, i_round)
        else:
            dones.append(done)
            step_cts.append(step_ct)
        
        return_list.append(episode_return)
        for i in range(n_group):
            episode_return[i].append(sum(mean_rewards[i]) / len(mean_rewards[i]))
        if use_wandb:
            import wandb
            wandb.log({'R/pred': episode_return[0][-1],'R/prey': episode_return[1][-1]})
        
        if train and save_every and (i_round + 1) % save_every == 0:
            print(Color.INFO.format('[INFO] Saving model ...'))
            agent.save(model_dir + '-predator', i_round)
            # agent.save(model_dir + '-predator', i_round)

            
    # return episode_return[0]
    return dones, step_cts


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
                
