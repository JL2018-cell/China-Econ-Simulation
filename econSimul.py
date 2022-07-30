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
INDUSTRIES_CHIN = ["农林牧渔业", "电力、热力、燃气及水生产和供应业", "金融业", "信息传输、软件和信息技术服务业",  "采矿业", "住宿和餐饮业", "制造业", "建筑业", "交通运输、仓储和邮政业", "批发和零售业", "教育业"]

def all_ones(x):
    return [1 for _ in x]

def empty_dicts(x):
    return [{} for _ in x]

def industry_weights(industries, weights):
    return dict(zip(industries, weights))

CO2_series, GDP_series, industry_dstr, industry_init_dstr, contribution = obtain_data(DATA_PATH)

growth = 0.2
sd = 1

contribution_chg_rate = {}
for metrics, provinces in contribution.items():
    contribution_chg_rate[metrics] = {}
    for province, attrs in provinces.items():
        contribution_chg_rate[metrics][province] = {}
        for k, v in attrs.items():
            contribution_chg_rate[metrics][province][k] = [growth, sd]

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
    'episode_length': 10, # Number of timesteps per episode
    'flatten_observations': False,
    'flatten_masks': False,

    'components': [
        #Build industries
        {"Construct": {"punishment": 0.5, "num_ep_to_recover": 5, "contribution": contribution}},
        #Exchange resources, industry points by auction.
        {'ContinuousDoubleAuction': {'max_num_orders': 5}},
    ],

    # Industries available in this world.
    # Help to define upper limit fo industries development,
    'industries': {industry: 2000 for industry in INDUSTRIES},
    'industries_chin': INDUSTRIES_CHIN,

    # (optional) kwargs of the chosen scenario class
    'starting_agent_resources': {"Food": 10., "Energy": 10.}, #food, energy
    'contribution': contribution,
    'contribution_chg_rate': contribution_chg_rate,
    'industry_depreciation': dict(zip(PROVINCES, [industry_weights(INDUSTRIES, all_ones(INDUSTRIES)) for prvn in PROVINCES])),
    'industry_weights': dict(zip(PROVINCES, [industry_weights(INDUSTRIES, all_ones(INDUSTRIES)) for prvn in PROVINCES])),
    'industry_init_dstr': industry_init_dstr,
    'dense_log_frequency': 1,
    # Use inverse reinforcement learning to know rewaed function of each agent.
    'irl': True,
    'irl_data_path': './data',
}

#for obj in env_config['contribution'].keys():
#env_config['contribution'][obj]

env = foundation.make_env_instance(**env_config)

obs = env.reset()

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


