import numpy as np
from multiagent.core import World, Agent, Landmark, Evader
from multiagent.scenario import BaseScenario


# >>>>>>>>>>>>> Kalman Filter >>>>>>>>>>>>>>>>
from filterpy.kalman import KalmanFilter
def create_kalman_filter(proc_model): # , dt, process_noise, measurement_noise, initial_state_covariance
    # F: State transition matrix for constant velocity model
    # H: Measurement matrix
    # Q: Process noise covariance matrix
    # R: Measurement noise covariance matrix
    # P: Initial state covariance matrix
    # x: Initial state (position and velocity)
    if proc_model == 'cv':
        return kalman_filter_rw()
    elif proc_model == 'rw':
        return kalman_filter_cv()
    else:
        raise NotImplementedError('Proc model {} not implemented!'.format(proc_model))
    
# process model: Random walk
def kalman_filter_rw():
    process_noise = 0.1
    measurement_noise = 1.0
    measurement_noise = 1e-3 # setting R to zero can cause numerical issues, so it is common to set it to a very small positive value.
    initial_state_covariance = np.eye(4)
    
    kf = KalmanFilter (dim_x=2, dim_z=2)
    kf.F = np.array([[1.,0.],
                    [0.,1.]])
    kf.H = np.array([[1.,0.],
                    [0.,1.]])
    # TODO: z = (x,y)-(x_0,y_0)
    kf.Q = process_noise * np.eye(2)
    kf.R = np.eye(2) * measurement_noise
    kf.P = np.eye(2) * 1000  # initial_state_covariance
    kf.x = np.zeros(2)        
    return kf

# process model: Constant velocity
def kalman_filter_cv():
    dt = 0.1  # Time step
    process_noise = 0.1
    measurement_noise = 1.0
    initial_state_covariance = np.eye(4)

    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])    
    kf.Q = process_noise * np.eye(4)
    kf.R = np.eye(2) * measurement_noise
    kf.P = initial_state_covariance
    kf.x = np.zeros(4)
    return kf

# <<<<<<<<<<<<< Kalman Filter <<<<<<<<<<<<<<<<        

class Scenario(BaseScenario):
    def make_world(self, num_adversaries=30, num_good_agents=10, noisy_obs=False, use_kf_act=False, kf_proc_model=None):
        world = World()
        # set any world properties first
        world.dim_c = 2
        # num_good_agents = 10
        # num_adversaries = 30
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 0   # 20
        self.noisy_obs = noisy_obs
        self.use_kf_act = use_kf_act
        # add agents
        world.agents = [Agent() for i in range(num_adversaries)] + [Evader() for i in range(num_good_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            if use_kf_act:
                agent.id = i
                assert kf_proc_model is not None, "kf_proc_model must be specified if use_kf_act"
                agent.kf = create_kalman_filter(kf_proc_model)
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.04
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            if isinstance(agent, Evader):
                _x, _y = agent.sample_point_on_path()
                # agent.state.p_pos = np.array([agent.x1,agent.y1])
                agent.state.p_pos = np.array([_x, _y])
                
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0
    def benchmark_data(self, agent, world):
        # return relative position observation of agent's opponents
        # NOTE the return is a subset of self.observation()'s return, so remember to stay constant with it
        observers = self.good_agents(world) if agent.adversary else self.adversaries(world)
        # ret = [None for _ in range(len(observers))]
        ret = []
        for observer in observers:
            delta_pos = agent.state.p_pos - observer.state.p_pos
            if self.noisy_obs:
                delta_pos_noisy = self._noisy_rel_pos(delta_pos)
                z = delta_pos_noisy
            else:
                z = delta_pos
            # z = np.concatenate([delta_pos_noisy] + [agent.state.p_vel])    # TODO how is vel observed?
            ret.append(z)
            # ret.append(agent.state.p_pos - observer.state.p_pos)
        return ret

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # # Agents are rewarded based on minimum agent distance to each landmark
        # main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        main_reward = self.zero_sum_reward(agent, world)
        return main_reward
    
    def zero_sum_reward(self, agent, world):
        rew = 0
        group_pred = self.adversaries(world)
        group_prey = self.good_agents(world)
        allies = group_pred if agent.adversary else group_prey
        opponents = group_prey if agent.adversary else group_pred
        if agent.collide:
            # print('haha! using new zero sum reward')
            for a in opponents:
                if self.is_collision(a, agent):
                    rew += 10 if agent.adversary else -10
        return rew
                    
    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                if self.is_collision(ag, agent):
                        rew += 10
            # for ag in agents:
                # for adv in adversaries:
                    # if self.is_collision(ag, adv):
                        # rew += 10
        return rew

    def _noisy_rel_pos(self, delta_pos):
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        noise_mean = 0
        noise_std_dev = 1.0 * dist
        # noise_std_dev = 0.1 * dist
        noise = np.random.normal(noise_mean, noise_std_dev, size=delta_pos.shape)
        delta_pos_noisy = delta_pos + noise
        return delta_pos_noisy
    
    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            delta_pos = other.state.p_pos - agent.state.p_pos
            if self.noisy_obs:
                # >>> Add noise to other_pos of opponents only >>>
                if other.adversary == agent.adversary:  # allies
                    other_pos.append(delta_pos)
                else:   # opponents
                    delta_pos_noisy = self._noisy_rel_pos(delta_pos)
                    other_pos.append(delta_pos_noisy)
                # <<< Add noise to other_pos of opponents only <<<
            else:
                other_pos.append(delta_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)
