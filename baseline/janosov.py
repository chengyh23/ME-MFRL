"""
Janosov, Milán, et al. "Group chasing tactics: how to catch a faster prey." New Journal of Physics 19.5 (2017): 053003.
"""
import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
# from examples.my_model.scenario_my1 import _get_act_preys
from examples.my_model.scenario_my1 import ModelEvader
BASE_DIR = '/home/chengyh23/Documents/ME-MFRL'

def cont2discrete_act(cont_act):
    discrete2cont = [
        [0, 0],
        [-1, 0],
        [+1, 0],
        [0, -1],
        [0, +1]
    ]
    dot_products = [np.dot(cont_act, unit_vec) for unit_vec in discrete2cont]
    discrete_act = np.argmax(dot_products)
    return discrete_act

class ModelPursuer:
    def __init__(self, algo_name, env):
        self.algo_name = algo_name
        self.env = env
    def act(self, obs):
        raise NotImplementedError()
    
class ModelPursuerJanosov(ModelPursuer):
    def __init__(self, algo_name, env):
        super(ModelPursuerJanosov, self).__init__(algo_name, env)
        # TODO inconsistent
        self.VMAX_C = 1  # Maximum speed of chasers

    # Function to calculate prediction of escaper's position
    def predict_position(self, escaper_pos, escaper_vel):
        VMAX_E = 2
        TAU = 1  # Prediction time
        escaper_vel_norm = np.linalg.norm(escaper_vel)
        if escaper_vel_norm == 0:
            return escaper_pos
        return escaper_pos + (escaper_vel * (VMAX_E / escaper_vel_norm)) * TAU
    
    # Function to calculate repulsion between chasers
    def repulsion_force(self, chaser1_pos, chaser2_pos):
        R_REPULSI = 0.2   # Interaction distance between chasers
        C_REPULSI = 0.5  # Interaction strength between chasers
        dist = np.linalg.norm(chaser1_pos - chaser2_pos)
        if dist < R_REPULSI:
            force_magnitude = C_REPULSI * self.VMAX_C / dist**2
            return force_magnitude * (chaser1_pos - chaser2_pos) / dist
        else:
            return np.zeros(2)
    
    def act(self, obs):
        chasers = [agent for agent in self.env.agents if agent.adversary==True]
        escapers = [agent for agent in self.env.agents if agent.adversary==False]
        
        # Keep consistent with the environment
        chaser_velocities = np.zeros((len(chasers), 2))
        
        for i, chaser in enumerate(chasers):
            _single_obs = obs[i]
            # chaser_pos = chaser.state.p_pos
            chaser_pos = _single_obs[2:4]
            
            N = len(escapers)
            escaper_positions = np.zeros((N,2))
            escaper_velocities = np.zeros((N,2))
            for j, escaper in enumerate(escapers):
                # _start = -1*2*(N-j) + len(_single_obs)
                # _end = -1*2*(N-j-1) + len(_single_obs)
                # escaper_positions[j] = _single_obs[_start:_end] + chaser_pos
                # escaper_velocities[j] = escaper.state.p_vel
                escaper_positions[j] = escaper.kf.x[:2]
                escaper_velocities[j] = escaper.kf.x[2:]
                
            closest_escaper_index = np.argmin(np.linalg.norm(escaper_positions - chaser_pos, axis=1))
            closest_escaper_pos, closest_escaper_vel = escaper_positions[closest_escaper_index], escaper_velocities[closest_escaper_index]
            predicted_pos = self.predict_position(closest_escaper_pos, closest_escaper_vel)
            # print(predicted_pos, escaper_positions[0], escapers[0].state.p_pos)
            
            # Chaser's desired velocity towards the predicted position
            chaser_desired_vel = (predicted_pos - chaser_pos) * (self.VMAX_C / np.linalg.norm(predicted_pos - chaser_pos))
            chaser_desired_vel *= 10.0
            # print('TARGET', chaser_desired_vel)
            
            # # Add repulsion from other chasers
            # for k, chaser_k in enumerate(chasers):
            #     if k==i: continue
            #     chaser_desired_vel += self.repulsion_force(chaser_pos, chaser_k.state.p_pos)
            #     # print(k, 'REPULSI', self.repulsion_force(chaser_pos, chaser_k.state.p_pos))
            
            # Update chaser's velocity (simple Euler integration)
            chaser_velocities[i] += chaser_desired_vel
            # chaser_positions[i] += chaser_velocities[i]
        # Continuous to Discrete Action
        
        acts = np.zeros((len(chasers),),dtype=np.int32)
        for i, chaser in enumerate(chasers):        
            acts[i] = cont2discrete_act(chaser_velocities[i])
        return acts

# # Parameters (some are arbitrary and for demonstration purposes)
# rcd = 0.5   # Catching distance
# rsens = 10  # Sensitivity range of escapers

# panic_threshold = 0.5  # Panic threshold for escaper zigzagging

np_random_seed = 42
np.random.seed(np_random_seed)

# # Initialize positions and velocities
# num_chasers = 3
# num_escapers = 1
# chaser_positions = np.random.uniform(-5, 5, (num_chasers, 2))
# escaper_positions = np.random.uniform(-5, 5, (num_escapers, 2))
# chaser_velocities = np.zeros((num_chasers, 2))
# escaper_velocities = np.ones((num_escapers, 2))



def fig_to_array(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    array = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    return array



class Runner2(object):
    def __init__(self, env, handles, map_size, max_steps, models, 
                play_handle, render_every=None, save_every=None, tau=None, log_name=None, log_dir=None, model_dir=None, render_dir=None, train=False, use_moment=True, use_kf_act=False, use_wandb=False):
        """Initialize runner

        Parameters
        ----------
        env: magent.GridWorld
            environment handle
        handles: list
            group handles
        map_size: int
            map size of grid world
        max_steps: int
            the maximum of stages in a episode
        render_every: int
            render environment interval
        save_every: int
            states the interval of evaluation for self-play update
        play_handle: method like
            run game
        tau: float
            tau index for self-play update
        log_name: str
            define the name of log dir
        log_dir: str
            donates the directory of logs
        
        """
        self.env = env
        self.max_steps = max_steps
        self.models = models
        self.handles = handles
        self.map_size = map_size
        self.render_every = render_every
        self.save_every = save_every
        self.play = play_handle
        self.render_dir = render_dir
        # self.train = train
        self.use_kf_act = use_kf_act
        self.use_wandb = use_wandb

        # if self.train:
        #     self.summary = SummaryObj(log_name=log_name, log_dir=log_dir)

        #     summary_items = ['ave_agent_reward', 'mean_reward', 'kill', "Sum_Reward", "Kill_Sum", "step_ct"]
        #     self.summary.register(summary_items)  # summary register
        #     self.summary_items = summary_items


    def run(self, iteration, mean_reward=None):
        # pass
        info = {'predator': {'mean_reward': 0.},
                'prey': {'mean_reward': 0.}}

        render = (iteration + 1) % self.render_every == 0 if self.render_every > 0 else False
        # render = render and not self.train    # only render when testing
        if render:
            print(f'Render @iter{iteration} -> {self.render_dir}')
        mean_rewards, done, step_ct = self.play(env=self.env, n_round=iteration, map_size=self.map_size, max_steps=self.max_steps, handles=self.handles, models=self.models, 
                    print_every=50, render=render, render_dir=self.render_dir, use_kf_act=self.use_kf_act)

        for i, tag in enumerate(['predator', 'prey']):
            info[tag]['mean_reward'] = mean_rewards[i]

        # # Change keys for logging both main and opponent
        # log_info = dict()
        # for key, value in info.items():
        #     log_info.update({key + 'tot_rew': value['mean_reward']})
        
        # # Success rate & average steps
        # self.summary.write({'kill': float(done), 'step_ct': step_ct}, iteration)
        
        # print('\n[INFO] {}'.format(info))
        # if self.use_wandb:
        #     import wandb
        #     wandb.log({'R/pred': info['predator']['mean_reward'],'R/prey': info['prey']['mean_reward']})
            
        # mean_reward['predator'].append(info['predator']['mean_reward'])
        # mean_reward['prey'].append(info['prey']['mean_reward'])
        # print('\n[INFO] {0}'.format(info))
        return done, step_ct



def play2(env, n_round, map_size, max_steps, handles, models, print_every=10, record=False, render=False, render_dir=None, train=False, use_kf_act=False):
    env.reset()

    step_ct = 0
    done = False
    n_group = 2

    rewards = [None for _ in range(n_group)]
    num_pred = 0
    num_prey = 0
    for agent in env.agents:
        if agent.adversary: num_pred += 1
        else: num_prey += 1
    max_nums = [num_pred, num_prey]  # num_pred predators, 40 prey


    # action_dim = [env.action_space[0].shape[0], env.action_space[-1].shape[0]]
    action_n = [env.action_space[0].n, env.action_space[-1].n]

    all_obs = env.reset()
    obs = [all_obs[:num_pred], all_obs[num_pred:]]  # gym-style: return first observation
    acts = [np.zeros((max_nums[i],), dtype=np.float32) for i in range(n_group)]
    values = [np.zeros((max_nums[i],), dtype=np.int32) for i in range(n_group)]
    logprobs = [np.zeros((max_nums[i],), dtype=np.int32) for i in range(n_group)]

    print("\n\n[*] ROUND #{0}, NUMBER: {1}".format(n_round, max_nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    former_act_prob = [np.zeros((1, env.action_space[0].n)), np.zeros((1, env.action_space[-1].n))]
    
    if render:
        obs_list = []
        obs_list.append(np.transpose(env.render(mode='rgb_array')[0], axes=(1, 0, 2)))
    ########################
    # Actor start sampling #
    ########################
    while not done and step_ct < max_steps:
        #################
        # Choose action #
        #################
        # print('\n===============obs len: ', len(obs))
        
        for i in range(n_group):
            if i==0:    # pred acts 
                acts[i] = models[0].act(obs[0])
            elif i==1:  # prey acts following fixed paths
                # preys = [agent for agent in env.agents if agent.adversary==False]
                acts[1] = models[1].act(obs[1])
        
        old_obs = obs
        stack_act = np.concatenate(acts, axis=0)
        all_obs, all_rewards, all_done, all_info = env.step(stack_act)
        obs = [all_obs[:num_pred], all_obs[num_pred:]]
        rewards = [all_rewards[:num_pred], all_rewards[num_pred:]]
        # done = all(all_done)
        done = all(all_done[num_pred:])
        stack_act = [stack_act[:num_pred], stack_act[num_pred:]]
        
        # >>>>>>>>>>>>> Kalman Filter >>>>>>>>>>>>>>>>
        # print(all_info)
        if use_kf_act:
            from examples.my_model.scenario_my1 import kf_step
            kf_step(env, all_info, num_pred)
        # <<<<<<<<<<<<< Kalman Filter <<<<<<<<<<<<<<<<        
        
        if render:
            obs_list.append(np.transpose(env.render(mode='rgb_array')[0], axes=(1, 0, 2)))
            
        for i in range(n_group):
            sum_reward = sum(rewards[i])
            mean_rewards[i].append(sum_reward / max_nums[i])
            total_rewards[i].append(sum_reward)
            

        info = {"kill": sum(total_rewards[0])/10}

        step_ct += 1

        if done: print("> step #{}, done!!!".format(step_ct))
        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))
    

    if render:
        from examples.my_model.scenario_my1 import render_gif
        render_gif(render_dir, obs_list, n_round)
        
    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        # mean_rewards[i] = sum(total_rewards[i])/max_nums[i]

    return mean_rewards, done, step_ct

def name2dir(_name, args):
    out_name = _name.replace('/', '-')
    log_dir = os.path.join(BASE_DIR,'data/tmp/{}'.format(out_name))
    model_dir = os.path.join(BASE_DIR, 'data/models/{}'.format(out_name))
    if args.test:
        render_dir = os.path.join(BASE_DIR, 'data/render/{}'.format(out_name))
    else:
        render_dir = os.path.join(BASE_DIR, 'data/render/{}'.format(out_name))

    if args.algo == 'me_mfq':
        log_dir = os.path.join(BASE_DIR,f'data/tmp/{args.algo}_{args.order}_{args.seed}')
        model_dir = os.path.join(BASE_DIR, f'data/models/{args.algo}_{args.order}_{args.seed}')
    return log_dir, model_dir, render_dir
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    
if __name__ == '__main__':
    from env.mpe.make_env import make_env
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy_obs', action='store_true', help='add noise to observation')   # 
    parser.add_argument('--noisy_factor', type=int, default=1, help='magnitude of noise added to relative obsrvation')
    parser.add_argument('--use_kf_act', action='store_true', help='maintain KF and use it to guide action selection')   # 1) maintain a KF for each agent, 2) update KF at each step, 3) ValueNet select action guided by KF, 
    parser.add_argument('--kf_proc_model', type=str, default='cv', help='KF Process model')
    parser.add_argument('--algo', type=str, default='janosov', help='choose an algorithm from the preset')
    parser.add_argument('--n_round', type=int, default=500, help='set the trainning round')
    parser.add_argument('--max_steps', type=int, default=400, help='set the max steps')
    parser.add_argument('--num_adversaries', type=int, default=3, help='number of predators')
    parser.add_argument('--num_good_agents', type=int, default=1, help='number of preys')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--map_size', type=int, default=40, help='set the size of map')  # then the amount of agents is 64
    parser.add_argument('--render', action='store_true', help='[for train] render or not (if true, will render every save)')
    parser.add_argument('--render_every', type=int, default=50, help='decide the render interval')
    parser.add_argument('--use_wandb', action='store_true', help='log onto wandb or not')
    # test
    parser.add_argument('--test', action='store_true', help='train (default) or test')
    parser.add_argument('--idx', nargs='*', help="required=True if testing")
    parser.add_argument('--test_n_round', type=int, default=100, help='set the testing round')
    parser.add_argument('--test_max_steps', type=int, default=100, help='set the test max steps')
    
    args = parser.parse_args()
    # >>>>>>>>> Default Setting
    args.test = True
    # args.test = True
    args.use_kf_act = True
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    _name = "se-R10"    # silly evader (fixed behavior)
    _name += f"/no{args.noisy_factor}" if args.noisy_obs else "/ao" # noiobs OR accobs
    _name += f"/{args.algo}_{args.test_n_round}x{args.test_max_steps}/{args.num_adversaries}v{args.num_good_agents}/{args.seed}"
    print("[CONFIGURATION]", _name)
    
    set_seed(args.seed)
    if not (args.use_kf_act and args.kf_proc_model=='cv'):
        raise Exception("in Janosov you need to use KF and constant velocity model~")
    env = make_env('exp_tag1',num_adversaries=args.num_adversaries, num_good_agents=args.num_good_agents, \
        noisy_obs=args.noisy_obs, noisy_factor=args.noisy_factor,use_kf_act=args.use_kf_act, kf_proc_model=args.kf_proc_model, \
        discrete_action_space=True, discrete_action_input=True, benchmark=True)
    log_dir, model_dir, render_dir = name2dir(_name, args)
    print("\n\nOutput to {} and {}".format(log_dir, model_dir))
    
    models = [ModelPursuerJanosov('janosov', env),
              ModelEvader('repulsive', env)]
    
    if not args.test:   # train
        print("Hey, Janosov don't have to be trained.")
    elif args.test:
        # assert args.idx is not None
        if not args.render:
            pass
            # raise Exception("Sure that do not render when testing?")

        runner = Runner2(env, None, args.map_size, args.test_max_steps, models, play2, 
                                render_every=args.render_every if args.render else 0, render_dir=render_dir, train=False, use_kf_act=args.use_kf_act)
        reward_ls = {'predator': [], 'prey': []}
        
        print("\n\n=== [TEST] {0} ROUNDs x {1} STEPs ===".format(args.test_n_round, args.test_max_steps))
        dones = []
        step_cts = []
        for k in range(0, args.test_n_round):
            done, step_ct = runner.run(k, mean_reward=reward_ls)
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
    

def gpt_version():
    # Simulation loop
    frames = 100
    images = []
    fig = plt.figure()
    for t in range(frames):
        # Update escaper's velocity with panic behavior (zigzagging)
        for i, (escaper_pos, escaper_vel) in enumerate(escaper_positions):
            min_dist_to_chaser = np.min(np.linalg.norm(chaser_positions - escaper_pos, axis=1))
            if min_dist_to_chaser < rsens and np.random.rand() < panic_threshold:
                # Implement zigzag behavior (simplified)
                escaper_vel += np.random.uniform(-1, 1, 2) * vmax_e
            else:
                escaper_vel *= 0  # Stop zigzagging, continue with direct escape

        # Update chaser's velocity towards the predicted position of the closest escaper
        for i, chaser_pos in enumerate(chaser_positions):
            closest_escaper_index = np.argmin(np.linalg.norm(escaper_positions - chaser_pos, axis=1))
            closest_escaper_pos, closest_escaper_vel = escaper_positions[closest_escaper_index], escaper_velocities[closest_escaper_index]
            predicted_pos = predict_position(closest_escaper_pos, closest_escaper_vel, vmax_e, tau_pred)
            
            # Chaser's desired velocity towards the predicted position
            chaser_desired_vel = (predicted_pos - chaser_pos) * (vmax_c / np.linalg.norm(predicted_pos - chaser_pos))
            
            # Add repulsion from other chasers
            for other_chaser_pos in chaser_positions[:i]:
                chaser_desired_vel += repulsion_force(chaser_pos, other_chaser_pos, rinter, Cinter, vmax_c)
            
            # Update chaser's velocity (simple Euler integration)
            chaser_velocities[i] += chaser_desired_vel
            chaser_positions[i] += chaser_velocities[i]
        
        # Simple boundary conditions (stay within a square)
        chaser_positions = np.clip(chaser_positions, -10, 10)
        escaper_positions = np.clip(escaper_positions, -10, 10)

        # Plotting (comment out or remove if not needed)
        plt.clf()
        plt.plot(chaser_positions[:, 0], chaser_positions[:, 1], 'bo', label='Chasers')
        plt.plot(escaper_positions[:, 0], escaper_positions[:, 1], 'rx', label='Escaper')
        plt.legend()
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.title("{}".format(t))
        # plt.pause(0.05)
        image = fig_to_array(fig)
        images.append(image)
    
    # plt.savefig('baseline/janosov.png', bbox_inches='tight', dpi=300)

    # plt.show()
    # Create the GIF using moviepy
    clip = ImageSequenceClip(images, fps=5)  # You can adjust the FPS as needed
    clip.write_gif("baseline/simulation.gif", fps=5)  # Write the GIF file
    print("GIF created successfully!")