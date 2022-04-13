from foundation.entities.resources import Resource, resource_registry
from foundation.base.base_component import BaseComponent, component_registry
from foundation.base.base_env import BaseEnvironment, scenario_registry
from foundation.base.base_agent import BaseAgent, agent_registry
from foundation.scenarios.MacroEcon.layout import MacroEconLayout
from foundation.components import component_registry
from foundation.components.Construct import Construct
from foundation.components.Transport import Transport
import foundation
import numpy as np




env_config = {
    'scenario_name': 'layout/MacroEcon',
    #to be contnued after layout construction on foundation/scenarios/MacroEcon.
    'world_size': [100, 100],
    'n_agents': 3,
    'agent_names': ["GuangDong", "HeBei", "XinJiang"],
    'agent_locs': [(80, 10), (50, 50), (10, 60)],
    'multi_action_mode_agents': True,
    'allow_observation_scaling': False,
    # Upper limit of industries that localGov can build per timestep.
    'buildUpLimit': 10,
    'episode_length': 10, # Number of timesteps per episode
    'flatten_observations': False,

    'components': [
        #Build industries
        {"Construct": {}},
        #Import resources from other provinces
        {"Transport": {}},
        #Exchange resources, industry points by auction.
        {'ContinuousDoubleAuction': {'max_num_orders': 5}},
    ],

    # Industries available in this world.
    'industries': ['Agriculture', 'Energy', 'Finance', \
                   'IT', 'Minerals', 'Tourism'], #Help to define actions of localGov

    # (optional) kwargs of the chosen scenario class
    'starting_agent_resources': {"Food": 10., "Energy": 10.} #food, energy
}

env = foundation.make_env_instance(**env_config)

obs = env.reset()

def sample_random_action(agent, mask):
    """Sample random UNMASKED action(s) for agent."""
    if agent.idx != 'p': #Not a planner
        actions_limit = ['Construct']
        limit_sum = sum([sum([act_lt in action_name for action_name in agent._action_names]) for act_lt in actions_limit])
    # Return a list of actions: 1 for each action subspace
    if agent.multi_action_mode:
        #Classify a 1D array to different actions.
        #'Construct.build_Agriculture', 'Construct.break_Agriculture', 'Construct.build_Energy',
        #'Construct.break_Energy', 'Construct.build_Finance', 'Construct.break_Finance', 
        #'Construct.build_IT', 'Construct.break_IT', 'Construct.build_Minerals',
        #'Construct.break_Minerals', 'Construct.build_Tourism', 'Construct.break_Tourism', 
        #'ContinuousDoubleAuction.Buy_Agriculture', 
        #'ContinuousDoubleAuction.Sell_Agriculture', 'ContinuousDoubleAuction.Buy_Minerals', 
        #'ContinuousDoubleAuction.Sell_Minerals', 'ContinuousDoubleAuction.Buy_Energy', 
        #'ContinuousDoubleAuction.Sell_Energy', 'ContinuousDoubleAuction.Buy_Tourism', 
        #'ContinuousDoubleAuction.Sell_Tourism', 'ContinuousDoubleAuction.Buy_IT', 
        #'ContinuousDoubleAuction.Sell_IT', 'ContinuousDoubleAuction.Buy_Finance', 
        #'ContinuousDoubleAuction.Sell_Finance'
        if agent.idx != 'p': #Not a planner
            temp_actions = [agent.buildUpLimit + 1]
            while sum(temp_actions[0:limit_sum]) > agent.buildUpLimit:
                split_masks = np.split(mask, agent.action_spaces.cumsum()[:-1])
                temp_actions = [np.random.choice(np.arange(len(m_)), p=m_/m_.sum()) for m_ in split_masks]
            return temp_actions
        else: #agent is centralGov/planner
            split_masks = np.split(mask, agent.action_spaces.cumsum()[:-1])
            return [np.random.choice(np.arange(len(m_)), p=m_/m_.sum()) for m_ in split_masks]

    # Return a single action
    else:
        return np.random.choice(np.arange(agent.action_spaces), p=mask/mask.sum())

def sample_random_actions(env, obs):
    """Samples random UNMASKED actions for each agent in obs."""
        
    actions = {
        #a_obs['action_mask'] defined in ai_ChinaEcon\foundation\base\base_env.py:702
        a_idx: sample_random_action(env.get_agent(a_idx), a_obs['action_mask'])
        for a_idx, a_obs in obs.items()
    }

    return actions

#Alternative: Write intelligent code to choose optimal actions.
actions = sample_random_actions(env, obs)
#call step to advance the state and advance time by one tick.
#This is 1 step only.
obs, rew, done, info = env.step(actions)
#Repeat until done.
while not done["__all__"]:
    obs, rew, done, info = env.step(actions)

print("Computation done.")
print("obs.keys:\n", obs.keys())
print("Investigate items.")
for key, val in obs['0'].items(): 
    print("{:50} {}".format(key, type(val)))
print("Reward of agents.")
for agent_idx, reward in rew.items(): 
    print("{:2} {:.3f}".format(agent_idx, reward))
print("Done")
print(done)

"""

def sample_random_actions(env, obs):
    #Samples random UNMASKED actions for each agent in obs.
        
    actions = {
        a_idx: sample_random_action(env.get_agent(a_idx), a_obs['action_mask'])
        for a_idx, a_obs in obs.items()
    }

    return actions

actions = sample_random_actions(env, obs)

obs, rew, done, info = env.step(actions)

"""
