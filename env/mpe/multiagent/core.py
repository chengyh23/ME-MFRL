import numpy as np

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
    
class Evader(Agent):
    def __init__(self):
        super(Evader, self).__init__()
        # change value of variable in the parent class
        self.collide = False
    def sample_initial_pos(self):
        raise NotImplementedError()
    def _update(self):
        raise NotImplementedError()
    def _get_act(self):
        raise NotImplementedError()
    
class EvaderRect(Evader):
    def __init__(self):
        super(EvaderRect, self).__init__()
        
        # new variable members
        self.direction = 2  # initilization
        # consistent with _set_action() in MultiAgentEnv
            # 0: (0, 0)
            # 1: (-1, 0)
            # 2: (+1, 0)
            # 3: (0, -1)
            # 4: (0, +1)
        # Rectangular path
        self.x1 = -1 + 0.2
        self.x2 = +1 - 0.2
        self.y1 = -1 + 0.2
        self.y2 = +1 - 0.2
        # self.step = 0.05    # TODO keep conssitent with mpe
        # # if a Evader dies, set self.movable = False
        # self.alive = True
        
    def sample_initial_pos(self):
        # Lengths of each side of the rectangle
        length_down = self.x2 - self.x1
        length_right = self.y2 - self.y1
        length_up = -(self.x1 - self.x2)
        length_left = -(self.y1 - self.y2)

        total_length = (length_right + length_down)*2
        _p = np.random.uniform(0, total_length)

        # Determine which side the point falls on
        distance_traveled = 0
        if _p < length_right:
            # On the right side
            x = self.x2
            y = self.y1 + (_p - distance_traveled)
        elif _p < length_right + length_down:
            # On the down side
            distance_traveled += length_right
            x = self.x2 - (_p - distance_traveled)
            y = self.y2
        elif _p < length_right + length_down + length_left:
            # On the left side
            distance_traveled += length_right + length_down
            x = self.x1
            y = self.y2 - (_p - distance_traveled)
        else:
            # On the up side
            distance_traveled += length_right + length_down + length_left
            x = self.x1 + (_p - distance_traveled)
            y = self.y1
        assert x<=self.x2 and x>=self.x1
        assert y<=self.y2 and y>=self.y1
        return (x, y)
    
    def _update(self):
        # called by world.step()
        # rect_dirs = ['right', 'down', 'left', 'up']
        #               2 ----> 4 ----> 1 ----> 3
        if not self.movable:
            return
        _pos = self.state.p_pos
        if self.direction == 2:
            if _pos[0] >= self.x2:
                self.direction = 4
        elif self.direction == 4:
            if _pos[1] >= self.y2:
                self.direction = 1
        elif self.direction == 1:
            if _pos[0] <= self.x1:
                self.direction = 3
        elif self.direction == 3:
            if _pos[1] <= self.y1:
                self.direction = 2
        else:
            raise ValueError("Invalid evader direction")
    def _get_act(self, discrete_action_space: bool, pursuers_positions=None):
        # consistent with _set_action() in MultiAgentEnv
            # 0: (0, 0)
            # 1: (-1, 0)
            # 2: (+1, 0)
            # 3: (0, -1)
            # 4: (0, +1)
        if not self.movable:
            return 0
        _action = self.direction   # 大方向
        thr = 0.1
        if self.direction == 2:
            if self.state.p_pos[1] < self.y1 - thr:
                _action = 4
            if self.state.p_pos[1] > self.y1 + thr:
                _action = 3
        elif self.direction == 4:
            if self.state.p_pos[0] < self.x2 - thr:
                _action = 2
            if self.state.p_pos[0] > self.x2 + thr:
                _action = 1
        elif self.direction == 1:
            if self.state.p_pos[1] < self.y2 - thr:
                _action = 4
            if self.state.p_pos[1] > self.y2 + thr:
                _action = 3
        elif self.direction == 3:
            if self.state.p_pos[0] < self.x1 - thr:
                _action = 2
            if self.state.p_pos[0] > self.x1 + thr:
                _action = 1
        # print(self.state.p_pos)
        # print(self.direction, action)
        if discrete_action_space:
            return _action
        else:
            discrete2cont = [
                [0, 0],
                [-1, 0],
                [+1, 0],
                [0, -1],
                [0, +1]
            ]
            action = discrete2cont[_action]
            return action
        
class EvaderRepulsive(Evader):
    def __init__(self):
        super(EvaderRepulsive, self).__init__()
    
    # Function to compute the repulsive force from a pursuer
    def compute_repulsive_force(self, pursuer_pos):
        C_PURSUER_REPULSION = 10.0  # Adjust this constant to tune repulsion strength

        evader_pos = self.state.p_pos
        direction = evader_pos - pursuer_pos
        distance = np.linalg.norm(direction)
        if distance == 0:
            return np.zeros_like(evader_pos)  # Avoid division by zero
        force_magnitude = C_PURSUER_REPULSION / (distance ** 2)  # Inverse-square law
        force = force_magnitude * direction / distance
        return force
    
    def compute_boundary_force(self):
        C_BOUNDARY_REPULSION = 1.0
        
        force = np.zeros_like(self.state.p_pos)
        xmin, xmax, ymin, ymax = -1, +1, -1, +1
        buffer_d = 0.3
        x, y = self.state.p_pos[0], self.state.p_pos[1]
        if x < xmin + buffer_d:
            force[0] += C_BOUNDARY_REPULSION / (x - xmin) ** 2
        elif x > xmax - buffer_d:
            force[0] -= C_BOUNDARY_REPULSION / (xmax - x) ** 2

        if y < ymin + buffer_d:
            force[1] += C_BOUNDARY_REPULSION / (y - ymin) ** 2
        elif y > ymax - buffer_d:
            force[1] -= C_BOUNDARY_REPULSION / (ymax - y) ** 2

        return force
    
    def _get_act(self, discrete_action_space: bool, pursuers_positions):
        total_force = np.zeros_like(self.state.p_pos)
        for pursuer_pos in pursuers_positions:
            total_force += self.compute_repulsive_force(pursuer_pos)
        total_force += self.compute_boundary_force()
        
        if discrete_action_space:
            from baseline.janosov import cont2discrete_act
            return cont2discrete_act(total_force)
        else:
            return total_force
    def _update(self):
        pass
    
    def sample_initial_pos(self):
        # Rectangular ring
        self.x1 = -1 + 0.2
        self.x2 = +1 - 0.2
        self.y1 = -1 + 0.2
        self.y2 = +1 - 0.2
        # Lengths of each side of the rectangle
        length_down = self.x2 - self.x1
        length_right = self.y2 - self.y1
        length_up = -(self.x1 - self.x2)
        length_left = -(self.y1 - self.y2)

        total_length = (length_right + length_down)*2
        _p = np.random.uniform(0, total_length)

        # Determine which side the point falls on
        distance_traveled = 0
        if _p < length_right:
            # On the right side
            x = self.x2
            y = self.y1 + (_p - distance_traveled)
        elif _p < length_right + length_down:
            # On the down side
            distance_traveled += length_right
            x = self.x2 - (_p - distance_traveled)
            y = self.y2
        elif _p < length_right + length_down + length_left:
            # On the left side
            distance_traveled += length_right + length_down
            x = self.x1
            y = self.y2 - (_p - distance_traveled)
        else:
            # On the up side
            distance_traveled += length_right + length_down + length_left
            x = self.x1 + (_p - distance_traveled)
            y = self.y1
        assert x<=self.x2 and x>=self.x1
        assert y<=self.y2 and y>=self.y1
        return (x, y)
# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)
            
        for agent in self.agents:
            if isinstance(agent, Evader):
                agent._update()

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise                
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                # print(a,b,'     ',f_a, f_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]