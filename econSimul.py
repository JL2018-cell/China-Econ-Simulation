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
import random

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
    'buildUpLimit': {'Agriculture': 10, 'Energy': 10},
    'episode_length': 10, # Number of timesteps per episode
    'flatten_observations': False,
    'flatten_masks': False,

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

"""
def sample_random_action(agent, mask):
    #Sample random UNMASKED action(s) for agent.
    # Return a list of actions: 1 for each action subspace
    if agent.multi_action_mode:
        actions = {k: [1 if random.random() > 0.5 else 0 for elm in v] for k, v in mask.items()}
        actions = agent.check_actions(actions, reset = False)
        split_masks = np.split(mask, agent.action_spaces.cumsum()[:-1])
        return [np.random.choice(np.arange(len(m_)), p=m_/m_.sum()) for m_ in split_masks]

    # Return a single action
    else:
        return np.random.choice(np.arange(agent.action_spaces), p=mask/mask.sum())
"""


def sample_random_action(agent, mask):
    """Sample random UNMASKED action(s) for agent."""

    # Return a list of actions: 1 for each action subspace
    if agent.multi_action_mode:
        #localGov takes 1 ~ 3 actions in a timestep.
        actions_num = random.randint(1, 3)
        actions = {}
        #Sample random actions
        for _ in range(actions_num):
            actions = {**actions, **agent.get_random_action()}
        return actions

    # Return a single action
    else:
        return agent.get_random_action()


def sample_random_actions(env, obs):
    """Samples random UNMASKED actions for each agent in obs."""
        
    actions = {
        #a_obs['action_mask'] defined in ai_ChinaEcon\foundation\base\base_env.py:702
        a_idx: sample_random_action(env.get_agent(a_idx), a_obs['action_mask'])
        for a_idx, a_obs in obs.items()
    }

    #Update actions of agent
    for a_idx, action in actions.items():
        if a_idx != 'p':
            env.world.agents[int(a_idx)].action.update(action)
        else:
            env.world.planner.action.update(action)

    return (env, actions)

#Alternative: Write intelligent code to choose optimal actions.
env, actions = sample_random_actions(env, obs)
#call step to advance the state and advance time by one tick.
#This is 1 step only.
obs, rew, done, info = env.step(actions)
#Repeat until done.
while not done["__all__"]:
    env, actions = sample_random_actions(env, obs)
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
