"""Self Play

```
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
from examples.my_model.scenario_my import play

from env.mpe.make_env import make_env

os.environ["WANDB_START_METHOD"] = "thread"

os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "4"

tf.disable_v2_behavior()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices={'attention_mfq', 'ac', 'mfac', 'mfq', 'dqn', 'me_mfq','me_mfq_leg','ppo','sac'}, help='choose an algorithm from the preset', required=True)
    parser.add_argument('--save_every', type=int, default=50, help='decide the self-play update interval')
    parser.add_argument('--update_every', type=int, default=5, help='decide the udpate interval for q-learning, optional')
    parser.add_argument('--n_round', type=int, default=500, help='set the trainning round')
    parser.add_argument('--render', action='store_true', help='render or not (if true, will render every save)')
    parser.add_argument('--map_size', type=int, default=40, help='set the size of map')  # then the amount of agents is 64
    parser.add_argument('--max_steps', type=int, default=400, help='set the max steps')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--order', type=int, default=4, help='moment order')
    parser.add_argument('--use_wandb', type=bool, default=False, help='log onto wandb or not')

    args = parser.parse_args()
    if args.use_wandb:
        import wandb
        wdb = wandb.init(project="MF-PE", resume="allow", name=f"myv5/{args.algo}_{args.n_round}x{args.max_steps}/{args.seed}")

    # set_seed(args.seed)
    # Initialize the environment    
    env = make_env('exp_tag',discrete_action_space=True, discrete_action_input=True)
    # handles = env.get_handles()   # 'MultiAgentEnv' object has no attribute 'get_handles'

    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True

    log_dir = os.path.join(BASE_DIR,'data/tmp/{}_{}'.format(args.algo, args.seed))
    model_dir = os.path.join(BASE_DIR, 'data/models/{}_{}'.format(args.algo, args.seed))

    if args.algo == 'me_mfq':
        log_dir = os.path.join(BASE_DIR,f'data/tmp/{args.algo}_{args.order}_{args.seed}')
        model_dir = os.path.join(BASE_DIR, f'data/models/{args.algo}_{args.order}_{args.seed}')

    if args.algo in ['mfq', 'mfac', 'attention_mfq','me_mfq']:
        use_mf = True
    else:
        use_mf = False

    start_from = 0

    sess = tf.Session(config=tf_config)
    models = [spawn_ai(args.algo, sess, env, None, args.algo + '-predator', args.max_steps, args.order),
              spawn_ai(args.algo, sess, env, None, args.algo + '-prey', args.max_steps, args.order)]

    sess.run(tf.global_variables_initializer())
    runner = tools.Runner(sess, env, None, args.map_size, args.max_steps, models, play,
                            render_every=args.save_every if args.render else 0, save_every=args.save_every, tau=0.01, log_name=args.algo,
                            log_dir=log_dir, model_dir=model_dir, train=True, use_wandb=args.use_wandb)

    print("\n\n=== {0} ROUNDs x {1} STEPs ===".format(args.n_round, args.max_steps))
    args.n_round
    for k in range(start_from, start_from + args.n_round):
        eps = linear_decay(k, [0, int(args.n_round * 0.8), args.n_round], [1, 0.2, 0.1])
        runner.run(eps, k)
