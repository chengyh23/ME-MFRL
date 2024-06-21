"""Self Play

```
python train_my.py --algo dqn --test --test_n_round 100 --render --render_every 30 --noisy_obs --use_kf_act  --kf_proc_model cv --idx 4999
python train_my.py --algo ppo --n_round 2000
```
"""

import argparse
import os
import random
import time

import tensorflow.compat.v1 as tf
import numpy as np
import magent
from examples.my_model.algo import spawn_ai
from examples.my_model.algo import tools
from examples.my_model.scenario_my1 import play

from env.mpe.make_env import make_env

os.environ["WANDB_START_METHOD"] = "thread"

os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "4"

tf.disable_v2_behavior()
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = '/home/chengyh23/Documents/ME-MFRL'


def set_seed(seed: int = 42) -> None:
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_random_seed(seed)
  tf.set_random_seed(seed)
  # When running on the CuDNN backend, two further options must be set
  os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
  os.environ['TF_DETERMINISTIC_OPS'] = '1'
  # Set a fixed value for the hash seed
  os.environ["PYTHONHASHSEED"] = str(seed)
  print(f"Random seed set as {seed}")

def linear_decay(epoch, x, y):
    min_v, max_v = y[0], y[-1]
    start, end = x[0], x[-1]

    if epoch == start:
        return min_v

    eps = min_v

    for i, x_i in enumerate(x):
        if epoch <= x_i:
            interval = (y[i] - y[i - 1]) / (x_i - x[i - 1])
            eps = interval * (epoch - x[i - 1]) + y[i - 1]
            break

    return eps

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
    parser.add_argument('--algo', type=str, choices={'attention_mfq', 'ac', 'mfac', 'mfq', 'dqn', 'me_mfq','me_mfq_leg','ppo','sac'}, help='choose an algorithm from the preset', required=True)
    parser.add_argument('--start_round', type=int, default=0, help='set the trainning round')
    parser.add_argument('--n_round', type=int, default=500, help='set the trainning round')
    parser.add_argument('--max_steps', type=int, default=400, help='set the max steps')
    parser.add_argument('--num_adversaries', type=int, default=3, help='number of predators')
    parser.add_argument('--num_good_agents', type=int, default=1, help='number of preys')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--map_size', type=int, default=40, help='set the size of map')  # then the amount of agents is 64
    parser.add_argument('--order', type=int, default=4, help='moment order')
    parser.add_argument('--render', action='store_true', help='[for train] render or not (if true, will render every save)')
    parser.add_argument('--render_every', type=int, default=50, help='decide the render interval')
    # train
    parser.add_argument('--save_every', type=int, default=50, help='decide the self-play update interval')
    parser.add_argument('--update_every', type=int, default=5, help='decide the udpate interval for q-learning, optional')
    parser.add_argument('--use_wandb', action='store_true', help='log onto wandb or not')
    # test
    # remember to run in: xvfb-run bash
    parser.add_argument('--test', action='store_true', help='train (default) or test')
    # parser.add_argument('--pred_dir', type=str, help='the path of the algorithm')
    parser.add_argument('--idx', nargs='*', help="required=True if testing")
    parser.add_argument('--test_n_round', type=int, default=100, help='set the testing round')
    parser.add_argument('--test_max_steps', type=int, default=100, help='set the test max steps')
    parser.add_argument('--test_noisy_factor', type=int, default=1, help='magnitude of noise added to relative obsrvation')
    # parser.add_argument('--test_noisy_factor', type=float, default=1, help='magnitude of noise added to relative obsrvation')
    
    args = parser.parse_args()
    # _name = ""
    _name = "re-Rsrspure"    # se: silly evader (fixed behavior) rect path, re: repulsive evader,
    _name += f"/no" if args.noisy_obs else "/ao" # noiobs OR accobs
    # _name += f"/no{args.noisy_factor}" if args.noisy_obs else "/ao" # noiobs OR accobs
    _name += f"/ka_{args.kf_proc_model}" if args.use_kf_act else "/eg"    # kfact OR epsgr
    # _name += f"/{args.algo}_{args.n_round}x{args.max_steps}/{args.num_adversaries}v{args.num_good_agents}/{args.seed}"
    _name += f"/{args.algo}_Xx{args.max_steps}/{args.num_adversaries}v{args.num_good_agents}/{args.seed}"
    # if args.use_kf_act:
    #     _name = f"kfv4_{args.kf_proc_model}/{args.algo}_{args.n_round}x{args.max_steps}/{args.num_adversaries}v{args.num_good_agents}/{args.seed}"
    # else:
    #     _name = f"myr1/{args.algo}_{args.n_round}x{args.max_steps}/{args.num_adversaries}v{args.num_good_agents}/{args.seed}"
    print("[CONFIGURATION]", _name)
    if args.use_wandb:
        import wandb
        wdb = wandb.init(project="MF-PE", resume="allow", name=_name)

    set_seed(args.seed)
    # # Initialize the environment    
    # env = make_env('exp_tag')
    # scenario_name = 'exppf_tag' if args.use_kf_act else 'exp_tag'
    env = make_env('exp_tag1',num_adversaries=args.num_adversaries, num_good_agents=args.num_good_agents, \
        noisy_obs=args.noisy_obs, noisy_factor=args.test_noisy_factor, use_kf_act=args.use_kf_act, kf_proc_model=args.kf_proc_model, \
        discrete_action_space=True, discrete_action_input=True, benchmark=True)
    # handles = env.get_handles()   # 'MultiAgentEnv' object has no attribute 'get_handles'

    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True

    # log_dir = os.path.join(BASE_DIR,'data/tmp/{}_{}'.format(args.algo, args.seed))
    # model_dir = os.path.join(BASE_DIR, 'data/models/{}_{}'.format(args.algo, args.seed))
    
    log_dir, model_dir, render_dir = name2dir(_name, args)
    print("\n\nOutput to {} and {}".format(log_dir, model_dir))

    if args.algo in ['mfq', 'mfac', 'attention_mfq','me_mfq']:
        use_mf = True
    else:
        use_mf = False


    sess = tf.Session(config=tf_config)
    from examples.my_model.scenario_my1 import ModelEvader
    models = [spawn_ai(args.algo, sess, env, None, args.algo + '-predator', args.max_steps, args.use_kf_act, args.order),
              ModelEvader('repulsive', env, args.use_kf_act)]
            #   None]
            #   spawn_ai(args.algo, sess, env, None, args.algo + '-prey', args.max_steps, args.use_kf_act, args.order)]

    sess.run(tf.global_variables_initializer())
    
    if not args.test:   # train
        if args.start_round > 0:
            models[0].load(model_dir + '-predator', step=args.start_round-1)
        # if not args.render_every == args.save_every:
        #     raise Warning("Attention: render_every != save_every")
        runner = tools.Runner(sess, env, None, args.map_size, args.max_steps, models, play,
                                render_every=args.render_every if args.render else 0, render_dir=render_dir, save_every=args.save_every, tau=0.01, log_name=args.algo,
                                log_dir=log_dir, model_dir=model_dir, train=True, use_kf_act=args.use_kf_act, use_wandb=args.use_wandb)
        print(f'\n\nnoisy_obs: {args.noisy_obs}, use_kf_act: {args.use_kf_act}, kf_proc_model: {args.kf_proc_model}')
        print("\n\n=== {0} ROUNDs x {1} STEPs ===".format(args.n_round, args.max_steps))
        for k in range(args.start_round, args.start_round + args.n_round):
            eps = linear_decay(k, [0, int(args.n_round * 0.8), args.n_round], [1, 0.2, 0.1])
            runner.run(eps, k)
    elif args.test:
        assert args.idx is not None
        if not args.render:
            # raise Exception("Sure that do not render when testing?")
            print("Sure that do not render when testing?")
        models[0].load(model_dir + '-predator', step=args.idx[0])

        runner = tools.Runner(sess, env, None, args.map_size, args.test_max_steps, models, play, 
                                render_every=args.render_every if args.render else 0, render_dir=render_dir)
        reward_ls = {'predator': [], 'prey': []}
        
        print("\n\n=== [TEST] {0} ROUNDs x {1} STEPs ===".format(args.test_n_round, args.test_max_steps))
        dones = []
        step_cts = []
        for k in range(0, args.test_n_round):
            done, step_ct = runner.run(0.0, k, mean_reward=reward_ls)
            dones.append(done)
            step_cts.append(step_ct)
            
        # print('\n[*] >>> Reward: Predator[{0}] max: {1}, min: {2}, std:{3}| Prey[{4}] max: {5}, min{6}, std:{7}'.format(
        #     args.pred, max(reward_ls['predator']), min(reward_ls['predator']), np.std(reward_ls['predator']),
        #     args.prey, max(reward_ls['prey']), min(reward_ls['prey']), np.std(reward_ls['prey'])))

        # print('\n[*] >>> Reward: Predator[{0}] {1} | Prey[{2} {3}]'.format(args.pred, sum(reward_ls['predator']) / args.test_n_round,
        #                                                                 args.prey, sum(reward_ls['prey']) / args.test_n_round))
        success_rate = np.mean(dones)
        avg_steps_ct = np.mean(step_cts)
        print("[CONFIGURATION]", _name)
        print(f"Success rate: {success_rate} | Avg steps {avg_steps_ct}")