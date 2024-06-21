import os
from env.mpe.make_env import make_env
# from stable_baselines3 import PPO, A2C
# from stable_baselines3.ppo import MlpPolicy
# env = make_env('exp_tag1')
# config = {"saveModel": "ppo",
#           "learning_rate": 0.0005,
#           "seed": 88,
#           "n_steps": 2048,
#           "batch_size": 128,
#           "tensorboard_log": "./tensorboard/",
#           "total_timesteps": 2_000
#           }

# model = A2C(
#     MlpPolicy,
#     env,
#     tensorboard_log=config["tensorboard_log"],
#     verbose=1,
#     device='cuda:5',
# )
# model.learn(total_timesteps=config["total_timesteps"])
import argparse
import random
import gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
from ddpg import DDPG

BASE_DIR = '/home/chengyh23/Documents/ME-MFRL'


def spawn_ai(algo):
    if not algo == 'ddpg':
        raise NotImplementedError()
    actor_lr = 3e-4
    critic_lr = 3e-3
    hidden_dim = 64
    gamma = 0.98
    tau = 0.005  # 软更新参数
    sigma = 0.01  # 高斯噪声标准差
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    # action_bound = env.action_space.high[0]  # 动作最大值
    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].shape[0]
    action_bound = env.action_space[0].high[0]  # 动作最大值
    
    device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
    
    agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)
    return agent

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # env.seed(0)
    torch.manual_seed(seed)

def name2dir(_name, args):
    out_name = _name.replace('/', '-')
    log_dir = os.path.join(BASE_DIR,'data/tmp/{}'.format(out_name))
    model_dir = os.path.join(BASE_DIR, 'data/models/{}'.format(out_name))
    if args.test:
        render_dir = os.path.join(BASE_DIR, 'data/render/{}/{}'.format(out_name, args.idx[0]))
    else:
        render_dir = os.path.join(BASE_DIR, 'data/render/{}'.format(out_name))

    if args.algo == 'me_mfq':
        log_dir = os.path.join(BASE_DIR,f'data/tmp/{args.algo}_{args.order}_{args.seed}')
        model_dir = os.path.join(BASE_DIR, f'data/models/{args.algo}_{args.order}_{args.seed}')
    return log_dir, model_dir, render_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy_obs', action='store_true', help='add noise to observation')   # 
    parser.add_argument('--noisy_factor', type=int, default=1, help='magnitude of noise added to relative obsrvation')
    parser.add_argument('--use_kf_act', action='store_true', help='maintain KF and use it to guide action selection')   # 1) maintain a KF for each agent, 2) update KF at each step, 3) ValueNet select action guided by KF, 
    parser.add_argument('--kf_proc_model', type=str, default='cv', help='KF Process model')
    parser.add_argument('--algo', type=str, default='ddpg', help='choose an algorithm from the preset')
    parser.add_argument('--n_round', type=int, default=500, help='set the trainning round')
    parser.add_argument('--max_steps', type=int, default=400, help='set the max steps')
    parser.add_argument('--num_adversaries', type=int, default=3, help='number of predators')
    parser.add_argument('--num_good_agents', type=int, default=1, help='number of preys')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--use_wandb', action='store_true', help='log onto wandb or not')
    # test
    parser.add_argument('--test', action='store_true', help='train (default) or test')
    parser.add_argument('--idx', nargs='*', help="required=True if testing")
    parser.add_argument('--test_n_round', type=int, default=100, help='set the testing round')
    parser.add_argument('--test_max_steps', type=int, default=100, help='set the test max steps')
    
    args = parser.parse_args()
    _name = "re"    # silly evader (fixed behavior)
    _name += f"/no{args.noisy_factor}" if args.noisy_obs else "/ao" # noiobs OR accobs
    _name += f"/{args.algo}_{args.n_round}x{args.max_steps}/{args.num_adversaries}v{args.num_good_agents}/{args.seed}"
    print("[CONFIGURATION]", _name)
    
    if args.use_wandb:
        import wandb
        wdb = wandb.init(project="MF-PE", resume="allow", name=_name)
        
    set_seed(args.seed)
    # # Initialize the environment    
    # env_name = 'Pendulum-v0'
    # env = gym.make(env_name)
    env_name = 'exp_tag1'
    # env = make_env(env_name)
    env = make_env('exp_tag1',num_adversaries=args.num_adversaries, num_good_agents=args.num_good_agents, \
        noisy_obs=args.noisy_obs, noisy_factor=args.noisy_factor,
        benchmark=True)
    
    log_dir, model_dir, render_dir = name2dir(_name, args)
    
    buffer_size = 10000
    # minimal_size = 1000
    minimal_size = 100
    batch_size = 64
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    from examples.my_model.scenario_my1 import ModelEvader
    agent = spawn_ai(args.algo)  # model
    models = [None,
              ModelEvader('repulsive', env, args.use_kf_act)]
    if not args.test:   # train
        return_list = rl_utils.train_off_policy_agent(
            env, agent, args.n_round, args.max_steps, replay_buffer, minimal_size, batch_size, 
            models, use_wandb=args.use_wandb, model_dir=model_dir, train=True, log_dir=log_dir)
    elif args.test: # test
        agent.load(model_dir + '-predator', step=args.idx[0])
        dones, step_cts = rl_utils.train_off_policy_agent(
            env, agent, args.test_n_round, args.test_max_steps, replay_buffer, minimal_size, batch_size, 
            train=False, log_dir=log_dir)
        
        success_rate = np.mean(dones)
        avg_steps_ct = np.mean(step_cts)
        print("[CONFIGURATION]", _name)
        print(f"Success rate: {success_rate} | Avg steps {avg_steps_ct}")
    # plot_returns(return_list)
    
def plot_returns(return_list):
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG on {}'.format(env_name))
    plt.savefig('baseline/ddpg.png', bbox_inches='tight', dpi=300)
    # plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG on {}'.format(env_name))
    plt.savefig('baseline/ddpg_smooth.png', bbox_inches='tight', dpi=300)
    # plt.show()