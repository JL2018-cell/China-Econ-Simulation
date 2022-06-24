from obtain_data import obtain_data
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
import matplotlib.pyplot as plt
import pickle

DATA_PATH = "./data"
PROVINCES = ["GuangDong", "HeBei", "XinJiang", "AnHui", "ZheJiang", "SiChuan", "FuJian", "HuBei", "JiangSu", "ShanDong", "HuNan", "HeNan", "ShanXi"]
INDUSTRIES = ['Agriculture', 'Energy', 'Finance', 'IT', 'Minerals', 'Tourism', 'Manufacturing', 'Construction', 'Transport', 'Retail', 'Education']

def all_ones(x):
    return [1 for _ in x]

def empty_dicts(x):
    return [{} for _ in x]

def industry_weights(industries, weights):
    return dict(zip(industries, weights))

industry_init_dstr, contribution = obtain_data(DATA_PATH)

env_config = {
    'scenario_name': 'layout/MacroEcon',
    #to be contnued after layout construction on foundation/scenarios/MacroEcon.
    'world_size': [100, 100],
    'n_agents': len(PROVINCES),
    'agent_names': PROVINCES,
    'agent_locs': [(i, i) for i in range(len(PROVINCES))],
    'multi_action_mode_agents': True,
    'allow_observation_scaling': False,
    # Upper limit of industries that localGov can build per timestep.
    'buildUpLimit': {'Agriculture': 10, 'Energy': 10},
    'episode_length': 2, # Number of timesteps per episode
    'flatten_observations': False,
    'flatten_masks': False,

    'components': [
        #Build industries
        {"Construct": {"punishment": 0.5, "num_ep_to_recover": 5}},
        #Exchange resources, industry points by auction.
        {'ContinuousDoubleAuction': {'max_num_orders': 5}},
    ],

    # Industries available in this world.
    # Help to define upper limit fo industries development,
    'industries': {industry: 2000 for industry in INDUSTRIES},

    # (optional) kwargs of the chosen scenario class
    'starting_agent_resources': {"Food": 10., "Energy": 10.}, #food, energy
    #'contribution': {"GDP": dict(zip(PROVINCES, dict(zip(INDUSTRIES, ALL_ONES)))),
    #                 "CO2": dict(zip(PROVINCES, dict(zip(INDUSTRIES, ALL_ONES)))),
    #                 "resource_points": dict(zip(PROVINCES, dict(zip(INDUSTRIES, ALL_ONES))))},
    #'contribution': {"GDP": dict(zip(PROVINCES, empty_dicts(PROVINCES))),
    #                 "CO2": dict(zip(PROVINCES, empty_dicts(PROVINCES))),
    #                 "resource_points": dict(zip(PROVINCES, empty_dicts(PROVINCES)))},
    'contribution': contribution,
    'industry_depreciation': dict(zip(PROVINCES, [industry_weights(INDUSTRIES, all_ones(INDUSTRIES)) for prvn in PROVINCES])),
    'industry_weights': dict(zip(PROVINCES, [industry_weights(INDUSTRIES, all_ones(INDUSTRIES)) for prvn in PROVINCES])),
    'industry_init_dstr': industry_init_dstr,
    'dense_log_frequency': 1
}

#for obj in env_config['contribution'].keys():
#env_config['contribution'][obj]

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
        actions = {action_name: 0 for action_name in agent._action_names}
        #Sample random actions
        for _ in range(actions_num):
            actions.update(agent.get_random_action())
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
actions = {k: [v for k, v in v.items()] for k, v in actions.items()}
#call step to advance the state and advance time by one tick.
#This is 1 step only.
obs, rew, done, info = env.step(actions)
#Repeat until done.
while not done["__all__"]:
    env, actions = sample_random_actions(env, obs)
    actions = {k: [v for k, v in v.items()] for k, v in actions.items()}
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

dense_logs = env.dense_log
with open('dense_logs_random.pkl', 'wb') as f:
    pickle.dump(dense_logs, f)
print("Save dense logs in dense_logs_random.pkl")

"""
#Plot graph
#Show location of agents
xs = [x for x, y in env_config['agent_locs']]
ys = [y for x, y in env_config['agent_locs']]

#Plot Geographical location of agents
fig, ax = plt.subplots()
ax.scatter(xs, ys)
for i, agent_name in enumerate(env_config['agent_names']):
    ax.annotate(agent_name, (xs[i], ys[i]))
#Pie chart of industry distribution for each agent: plt.pie(obs['0']['world-industries'].values(), labels = obs['0']['world-industries'].keys())
#plt.plot(xs, ys, "ro")
plt.show()

#Plot industry distribution
fig, axs = plt.subplots(2, env_config['n_agents'] // 2 + env_config['n_agents'] % 2)
for i, agent_idx in enumerate(obs.keys()):
    #obs[agent_idx]['world-industries']
    if agent_idx != 'p':
        axs[i // 2][i % 2].pie(obs[agent_idx]['world-industries'].values(), labels = obs['0']['world-industries'].keys())
plt.show()

#Plot actions taken.
fig, axs = plt.subplots(2, env_config['n_agents'] // 2 + env_config['n_agents'] % 2)
for i, agent_idx in enumerate(obs.keys()):
    if agent_idx != 'p':
        axs[i // 2][i % 2].hist(obs[agent_idx]['world-actions'])
plt.show()

"""

"""
#Histogram
for agent in obs.keys():
    plt.hist(obs[agent]['world-actions'])
    plt.show()
    plt.hist(obs[agent]['world-industries'])
    plt.show()


#Not useful
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
