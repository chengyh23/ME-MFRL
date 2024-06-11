"""
Pursuit: predators get reward when they attack prey.
"""

import argparse
import logging
import time
import logging as log
import numpy as np


from env.mpe.make_env import make_env

logging.basicConfig(level=logging.ERROR)

num_pred = 30
num_prey = 10

# order = 4
# moment_dim = order * 2
# bin_dim = order ** 2

def _calc_bin_density(a, order=4, action_max=1.0, action_min=-1.0, eps=1e-4):
    bin_density = np.zeros(shape=(order, order), dtype=np.float)
    action_tmp = (a - action_min) / (action_max - action_min) * (order - eps-4)
    action_tmp = np.floor(action_tmp).astype(np.int)

    for i in range(len(action_tmp)):
        bin_density[action_tmp[i][0], action_tmp[i][1]] += 1

    bin_density = bin_density.flatten()
    bin_density = bin_density / np.sum(bin_density)
    return bin_density


def _calc_moment(a, order=4):
    moments = []
    for i in range(1, order+1):
        moment_1 = np.mean(np.power(a[:,0], i))
        moment_1 = np.sign(moment_1) * np.power(np.abs(moment_1), 1/i)
        moment_2 = np.mean(np.power(a[:,1], i))
        moment_2 = np.sign(moment_2) * np.power(np.abs(moment_2), 1/i)
        moments.append(moment_1)
        moments.append(moment_2) 
    moment = np.array(moments)
    return moment.reshape(1,-1)

def _cal_ca(a, order=4):
    c_a = []
    for i in range(a.shape[1]):
        mean_action = np.mean(a[:,i])
        deviation = mean_action - a[:,i]
        m = np.array([j*a[:,i]**(j-1) for j in range(1,order+1)])
        c = m@deviation.T 
        c_a.append(c)

    c_a = np.array(c_a).T
    return c_a.reshape(1,-1)

def kf_step(env,all_info):
    # kf_system.predict()
    for agent in env.agents:
        agent.kf.predict()
        # Update kf using measurements from all observers of an agent
        for i, z in enumerate(all_info['n'][agent.id]):    # relative measurements from opponents
            # pos_rel = z[:2]
            # vel = z[2:]
            # observer_state = agent.kf.x[:2]
            observer_idx = (i + num_pred) if agent.adversary else i
            observer_pos = env.agents[observer_idx].state.p_pos
            # pos_absolute = pos_rel + observer_pos
            z[:2] += observer_pos
            agent.kf.update(z)
    belief = []
    # Calc estimation error
    errors = 0.0
    # errors_vel = 0.0
    for agent in env.agents:
        belief.append(agent.kf.x)
        error = np.sqrt(np.sum((agent.state.p_pos - agent.kf.x[:2])**2))
        errors += error
        # error_vel = np.sqrt(np.sum((agent.state.p_vel - agent.kf.x[2:])**2))
        # errors_vel += error_vel
    # print(belief)
    # print(errors)
    # print(errors, errors_vel)
    
    
    # # Uncertainty (Covariance)
    # all_cov = []
    # for agent in env.agents:
    #     all_cov.append(np.diag(agent.kf.Q))
    # pred_cov = np.concatenate(all_cov[:num_pred])
    # prey_cov = np.concatenate(all_cov[num_pred:])
    # # each agent use cov of its opponent agents
    # cov = [np.tile(prey_cov, (num_pred,1)), np.tile(pred_cov, (num_prey,1))]
    
def _get_act_preys(evaders):
    actions = np.zeros(len(evaders), dtype=np.int32)
    for i,evader in enumerate(evaders):
        actions[i] = evader._get_act()
    return actions

def play(env, n_round, map_size, max_steps, handles, models, print_every=10, record=False, render=False, render_dir=None, eps=None, train=False, use_kf_act=False):
    env.reset()

    step_ct = 0
    done = False
    n_group = 2

    rewards = [None for _ in range(n_group)]
    _num_pred = 0
    _num_prey = 0
    for agent in env.agents:
        if agent.adversary: _num_pred += 1
        else: _num_prey += 1
    #
    num_pred = _num_pred
    num_prey = _num_prey
    assert num_pred==_num_pred
    assert num_prey==_num_prey
    max_nums = [num_pred, num_prey]  # num_pred predators, 40 prey


    # action_dim = [env.action_space[0].shape[0], env.action_space[-1].shape[0]]
    action_n = [env.action_space[0].n, env.action_space[-1].n]

    all_obs = env.reset()
    obs = [all_obs[:num_pred], all_obs[num_pred:]]  # gym-style: return first observation
    acts = [np.zeros((max_nums[i],), dtype=np.int32) for i in range(n_group)]
    values = [np.zeros((max_nums[i],), dtype=np.int32) for i in range(n_group)]
    logprobs = [np.zeros((max_nums[i],), dtype=np.int32) for i in range(n_group)]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, max_nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    former_act_prob = [np.zeros((1, env.action_space[0].n)), np.zeros((1, env.action_space[-1].n))]
    # for i in range(n_group):
    #     if 'me' in models[i].name:
    #         former_g[i] = np.zeros((1, models[i].moment_dim))
    #     if 'mf' in models[i].name:
    #         former_meanaction[i] = np.zeros((1, models[i].moment_dim))
    #     elif 'ma' in models[i].name:
    #         former_meanaction[i] = np.zeros((1, max_nums[i]*models[i].num_actions))

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
            if i==0:    # pred acts by RL model
                # if 'mf' in models[i].name:
                #     former_meanaction[i] = np.tile(former_meanaction[i], (max_nums[i], 1))
                # if 'ma' in models[i].name:
                #     former_meanaction[i] = np.tile(former_meanaction[i], (max_nums[i], 1))
                # if 'me' in models[i].name:
                #     former_g[i] = np.tile(former_g[i], (max_nums[i], 1))
                former_act_prob[0] = np.tile(former_act_prob[0], (max_nums[0], 1))
                # kf_cov[i] = env.world.good_agents()
                acts[0], _, logprobs[0] = models[0].act(obs=obs[0], prob=former_act_prob[0], eps=eps, train=True)   # TODO only mean-field of allies? not considering that of opponents?
                # acts[i], _, logprobs[i] = models[i].act(obs=obs[i], prob=former_act_prob[i], eps=eps, train=True)   # Continuous Action space
            elif i==1:  # prey acts following fixed paths
                preys = [agent for agent in env.agents if agent.adversary==False]
                acts[1] = _get_act_preys(preys)
        
        old_obs = obs
        stack_act = np.concatenate(acts, axis=0)
        all_obs, all_rewards, all_done, all_info = env.step(stack_act)
        obs = [all_obs[:num_pred], all_obs[num_pred:]]
        rewards = [all_rewards[:num_pred], all_rewards[num_pred:]]
        done = all(all_done)
        if train==False:
            if done:
                raise Exception
        stack_act = [stack_act[:num_pred], stack_act[num_pred:]]
        
        # >>>>>>>>>>>>> Kalman Filter >>>>>>>>>>>>>>>>
        # print(all_info)
        if use_kf_act:
            kf_step(env, all_info)
        # <<<<<<<<<<<<< Kalman Filter <<<<<<<<<<<<<<<<        
        if train:
            predator_buffer = {
                'state': old_obs[0], 
                'actions': acts[0], 
                'rewards': rewards[0], 
                'dones': all_done[:num_pred],
                'values': values[0], 
                'logps': logprobs[0],
                'ids': range(max_nums[0]), 
            }
            # if 'me' in models[0].name:
            #     predator_buffer['g'] = former_g[0]
            # if 'mf' in models[0].name or 'ma' in models[0].name:
            #     predator_buffer['meanaction'] = former_meanaction[0]
            predator_buffer['prob'] = former_act_prob[0]
            if 'sac' in models[0].name:
                predator_buffer['next_state'] = obs[0]

            # prey_buffer = {
            #     'state': old_obs[1], 
            #     'actions': acts[1], 
            #     'rewards': rewards[1], 
            #     'dones': all_done[num_pred:],
            #     'values': values[1], 
            #     'logps': logprobs[1],
            #     'ids': range(max_nums[1]), 
            # }
            # # if 'me' in models[1].name:
            # #     prey_buffer['g'] = former_g[1]
            # # if 'mf' in models[1].name or 'ma' in models[1].name:
            # #     prey_buffer['meanaction'] = former_meanaction[1]
            # prey_buffer['prob'] = former_act_prob[1]
            # if 'sac' in models[1].name:
            #     prey_buffer['next_state'] = obs[1]

            models[0].flush_buffer(**predator_buffer)
            # models[1].flush_buffer(**prey_buffer)
        
        #############################
        # Calculate mean field #
        #############################
        for i in range(n_group):
            # if 'me' in models[i].name:
            #     former_g[i] = _cal_ca(acts[i])
            #     former_meanaction[i] = _calc_moment(acts[i])
            # if 'grid' in models[i].name:
            #     former_meanaction[i] = _calc_bin_density(acts[i])      
            # if 'ma' in models[i].name:
            #     former_meanaction[i] = stack_act[i].reshape(1,-1)
            former_act_prob[i] = np.mean(list(map(lambda x: np.eye(action_n[i])[x], acts[i])), axis=0, keepdims=True)

        if render:
            obs_list.append(np.transpose(env.render(mode='rgb_array')[0], axes=(1, 0, 2)))
            
        for i in range(n_group):
            sum_reward = sum(rewards[i])
            total_rewards[i].append(sum_reward)
            

        info = {"kill": sum(total_rewards[0])/10}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))
    if train:
        # if 'ppo' in models[0].name:
        #     predator_buffer = {
        #         'state': obs[0], 
        #         'acts': [None for i in range(max_nums[0])], 
        #         'rewards': [None for i in range(max_nums[0])], 
        #         'dones': [None for i in range(max_nums[0])],
        #         'values': [None for i in range(max_nums[0])], 
        #         'logps': [None for i in range(max_nums[0])],
        #         'ids': range(max_nums[0]), 
        #     }
        #     if 'me' in models[0].name:
        #         predator_buffer['g'] = np.tile(former_g[0], (max_nums[0], 1))
        #     if 'mf' in models[0].name or 'ma' in models[0].name:
        #         predator_buffer['meanaction'] = np.tile(former_meanaction[0], (max_nums[0], 1))
        
        #     models[0].flush_buffer(**predator_buffer)
        
        # if 'ppo' in models[1].name:
        #     prey_buffer = {
        #         'state': obs[1], 
        #         'acts': [None for i in range(max_nums[1])], 
        #         'rewards': [None for i in range(max_nums[1])], 
        #         'dones': [None for i in range(max_nums[1])],
        #         'values': [None for i in range(max_nums[1])], 
        #         'logps': [None for i in range(max_nums[1])],
        #         'ids': range(max_nums[1]), 
        #     }
        #     if 'me' in models[1].name:
        #         prey_buffer['g'] = np.tile(former_g[1], (max_nums[1], 1))
        #     if 'mf' in models[1].name or 'ma' in models[1].name:
        #         prey_buffer['meanaction'] = np.tile(former_meanaction[1], (max_nums[1], 1))

        #     models[1].flush_buffer(**prey_buffer)


        models[0].train()
        # models[1].train()

    if render:
        # render_outdir = 'data/render/{}-{}'.format(models[0].name, models[1].name)
        render_gif(render_dir, obs_list, n_round)
        
    for i in range(n_group):
        mean_rewards[i] = sum(total_rewards[i])/max_nums[i]

    return mean_rewards

def test(env, n_round, map_size, max_steps, handles, models, print_every=10, record=False, render=False, eps=None, train=False):
    env.reset()

    step_ct = 0
    done = False
    n_group = 2

    rewards = [None for _ in range(n_group)]
    max_nums = [num_pred, num_prey]

    # action_dim = [env.action_space[0].shape[0], env.action_space[-1].shape[0]]
    action_n = [env.action_space[0].n, env.action_space[-1].n]

    all_obs = env.reset()
    obs = [all_obs[:num_pred], all_obs[num_pred:]]  # gym-style: return first observation
    acts = [np.zeros((max_nums[i],), dtype=np.int32) for i in range(n_group)]
    values = [np.zeros((max_nums[i],), dtype=np.int32) for i in range(n_group)]
    logprobs = [np.zeros((max_nums[i],), dtype=np.int32) for i in range(n_group)]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, max_nums[0]))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    former_act_prob = [np.zeros((1, env.action_space[0].n)), np.zeros((1, env.action_space[-1].n))]
    # former_meanaction = [[] for _ in range(n_group)]
    # for i in range(n_group):
    #     if 'mf' in models[i].name:
    #         former_meanaction[i] = np.zeros((1, models[i].moment_dim))

    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import platform
    if platform.system() == "Linux":
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
    if render:
        obs_list = []
        # tmp = env.render(mode='rgb_array')
        # print('Render Out: ',type(tmp), len(tmp),tmp[0].shape)
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
            if i==0:    # pred
                # if 'mf' in models[i].name:
                #     former_meanaction[i] = np.tile(former_meanaction[i], (max_nums[i], 1))
                former_act_prob[i] = np.tile(former_act_prob[i], (max_nums[i], 1))
                acts[i], _ = models[i].act(obs=obs[i], prob=former_act_prob[i], eps=eps, train=True)
            elif i==1:  # prey
                # acts[i], values[i], logprobs[i] = models[i].act(state=obs[i], meanaction=former_meanaction[i])
                preys = [agent for agent in env.agents if agent.adversary==False]
                acts[1] = _get_act_preys(preys)
        ## random predator
        # acts[0] = np.random.rand(num_pred,2)*2-1  

        old_obs = obs
        stack_act = np.concatenate(acts, axis=0)
        all_obs, all_rewards, all_done, _ = env.step(stack_act)
        obs = [all_obs[:num_pred], all_obs[num_pred:]]
        rewards = [all_rewards[:num_pred], all_rewards[num_pred:]]
        done = all(all_done)
        if done:
            raise Exception
        #############################
        # Calculate mean field #
        #############################
        for i in range(n_group):
            # if 'grid' in models[i].name:
            #     former_meanaction[i] = _calc_bin_density(acts[i], order=models[i].order)
            # elif 'mf' in models[i].name:
            #     former_meanaction[i] = _calc_moment(acts[i], order=models[i].order)
            former_act_prob[i] = np.mean(list(map(lambda x: np.eye(action_n[i])[x], acts[i])), axis=0, keepdims=True)
            
        
        if render:
            # tmp = env.render(mode='rgb_array')
            # print('Render Out: ',type(tmp), len(tmp),tmp[0].shape)
            obs_list.append(np.transpose(env.render(mode='rgb_array')[0], axes=(1, 0, 2)))
            
        for i in range(n_group):
            sum_reward = sum(rewards[i])
            total_rewards[i].append(sum_reward)
        
        info = {"kill": sum(total_rewards[0])/10}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))
    
    if render:
        render_outdir = 'data/render/{}-{}'.format(models[0].name, models[1].name)
        render_gif(render_outdir, obs_list, n_round)
    
    for i in range(n_group):
        mean_rewards[i] = sum(total_rewards[i])/max_nums[i]

    return mean_rewards

def render_gif(render_outdir, obs_list, n_round):
    import os
    if not os.path.exists(render_outdir):
        os.makedirs(render_outdir)
    from moviepy.editor import ImageSequenceClip
    if len(obs_list) > 0:
        print('[*] Saving Render')
        clip = ImageSequenceClip(obs_list, fps=1)
        clip.write_gif('{}/replay_{}.gif'.format(render_outdir, n_round), fps=1, verbose=False)
        print('[*] Saved Render')