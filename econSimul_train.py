from obtain_data import obtain_data
import ray
from ray.rllib.agents.ppo import PPOTrainer
from foundation.entities.resources import Resource, resource_registry
from foundation.base.base_component import BaseComponent, component_registry
from foundation.base.base_env import BaseEnvironment, scenario_registry
from foundation.base.base_agent import BaseAgent, agent_registry
from foundation.scenarios.MacroEcon.layout import MacroEconLayout
from foundation.components import component_registry
from foundation.components.Construct import Construct
from foundation.components.continuous_double_auction import ContinuousDoubleAuction
from utils.env_wrapper import RLlibEnvWrapper
import foundation
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import copy
import json

DATA_PATH = "./data"
PROVINCES = ["GuangDong", "HeBei", "XinJiang", "AnHui", "ZheJiang", "SiChuan", "FuJian", "HuBei", "JiangSu", "ShanDong", "HuNan", "HeNan", "ShanXi"]
INDUSTRIES = ['Agriculture', 'Energy', 'Finance', 'IT', 'Minerals', 'Tourism', 'Manufacturing', 'Construction', 'Transport', 'Retail', 'Education']
INDUSTRIES_CHIN = ["农林牧渔业", "电力、热力、燃气及水生产和供应业", "金融业", "信息传输、软件和信息技术服务业",  "采矿业", "住宿和餐饮业", "制造业", "建筑业", "交通运输、仓储和邮政业", "批发和零售业", "教育业"]

def all_ones(x):
    return [1 for _ in x]

def all_zeros(x):
    return [1 for _ in x]

def random_ones(x):
    return [random.randint(0,1) for _ in x]

def empty_dicts(x):
    return [{} for _ in x]

def industry_weights(industries, weights):
    return dict(zip(industries, weights))

def compare_difference(simul, episode_length):
    hist_CO2, hist_GDP, hist_industry, _, _ = obtain_data(DATA_PATH, "2011年", "2021年")
    hist = {"GDP": {k: v[::-1] for k, v in hist_GDP.items()}, \
            "CO2": {k: v[::-1] for k, v in hist_CO2.items()}, \
            "industry": {k: v[::-1] for k, v in hist_industry.items()}}

    cumu_diff = 0
    simul_GDP = {}
    simul_CO2 = {}
    simul_industry = {}
    simul_ = {"GDP": {}, "CO2": {}, "industry": {}}
    # Iterate over agents
    # Calculate average
    for ep in simul.keys():
        for agent_idx in simul[ep]['states'][0].keys():
            if agent_idx != 'p':
                agent_name = simul[ep]['states'][0][agent_idx]['name']
                try:
                    simul_["GDP"][agent_name] += pd.Series([simul[ep]['states'][i + 1]['0']['endogenous']['GDP'] for i in range(episode_length)])
                except KeyError:
                    simul_["GDP"][agent_name] = pd.Series([simul[ep]['states'][i + 1]['0']['endogenous']['GDP'] for i in range(episode_length)])
                try:
                    simul_["CO2"][agent_name] += pd.Series([simul[ep]['states'][i + 1]['0']['endogenous']['CO2'] for i in range(episode_length)])
                except KeyError:
                    simul_["CO2"][agent_name] = pd.Series([simul[ep]['states'][i + 1]['0']['endogenous']['CO2'] for i in range(episode_length)])
                try:
                    simul_["industry"][agent_name] += pd.DataFrame([simul[ep]['states'][i + 1]['0']['inventory'] for i in range(episode_length)])
                except KeyError:
                    simul_["industry"][agent_name] = pd.DataFrame([simul[ep]['states'][i + 1]['0']['inventory'] for i in range(episode_length)])

    # Calculate difference between simulated data and historical data.
    diff = {"GDP": {}, "CO2": {}, "industry": {}}
    for province in simul_["GDP"].keys():
        simul_["GDP"][province].index = hist["GDP"][province].index
        diff["GDP"][province] = simul_["GDP"][province] - hist["GDP"][province]
    for province in simul_["CO2"].keys():
        simul_["CO2"][province].index = hist["CO2"][province].index
        diff["CO2"][province] = simul_["CO2"][province] - hist["CO2"][province]
    for province in simul_["industry"].keys():
        simul_["industry"][province].index = hist["industry"][province].index
        hist["industry"][province].columns = simul_["industry"][province].columns
        diff["industry"][province] = simul_["industry"][province] - hist["industry"][province]
    #diff["GDP"] = {k: [list(v[::-1].index), list(v[::-1].values)] for k, v in diff["GDP"].items()}
    #diff["CO2"] = {k: [list(v[::-1].index), list(v[::-1].values)] for k, v in diff["CO2"].items()}
    diff["GDP"] = {k: [list(v.index), list(v.values)] for k, v in diff["GDP"].items()}
    diff["CO2"] = {k: [list(v.index), list(v.values)] for k, v in diff["CO2"].items()}
    diff.pop("industry")
    return diff

def simplify_actions(dense_logs):
    # Every trial
    for tr in dense_logs.keys():
        # Every episode
        for ep in range(len(dense_logs[tr]['actions'])):
            # Every agents
            for agent_idx in dense_logs[tr]['actions'][ep].keys():
                if agent_idx != "p":
                    tmp = {}
                    actions = sorted(dense_logs[tr]['actions'][ep][agent_idx].keys())
                    low_q = len(actions) // 4 - 1
                    mid = len(actions) // 2 - 1
                    up_q = 3 * len(actions) // 4 - 1
                    end = len(actions) - 1
                    for i in range(11):
                        tmp[actions[mid - i].split(".")[-1]] = dense_logs[tr]['actions'][ep][agent_idx][actions[mid - i]] - dense_logs[tr]['actions'][ep][agent_idx][actions[low_q - i]]
                        tmp[actions[end - i].split(".")[-1]] = dense_logs[tr]['actions'][ep][agent_idx][actions[end - i]] - dense_logs[tr]['actions'][ep][agent_idx][actions[up_q - i]]
                dense_logs[tr]['actions'][ep][agent_idx] = tmp
    return dense_logs

CO2_series, GDP_series, industry_dstr, industry_init_dstr, contribution = obtain_data(DATA_PATH, None, "2011年")

contribution_chg_rate = {}
for metrics, provinces in contribution.items():
    contribution_chg_rate[metrics] = {}
    for province, attrs in provinces.items():
        contribution_chg_rate[metrics][province] = {}
        for k, v in attrs.items():
            if metrics == "CO2":
                contribution_chg_rate[metrics][province][k] = [0.9, 0.5]
            elif metrics == "resource_points":
                contribution_chg_rate[metrics][province][k] = [0.8, 0.5]
            else: # GDP
                contribution_chg_rate[metrics][province][k] = [0.5, 0.5]

env_config = {
    "scenario_name": 'layout/MacroEcon',

    'components': [
        #Build industries
        {"Construct": {"punishment": 0.5, "num_ep_to_recover": 5}},
        #Exchange resources, industry points by auction.
        {'ContinuousDoubleAuction': {'max_num_orders': 5}},
    ],

    # ===== SCENARIO CLASS ARGUMENTS =====
    'agent_names': PROVINCES,
    'agent_locs': [(i, i) for i in range(len(PROVINCES))],
    # Resource points that each agents van get in each time step.
    'buildUpLimit': {'Agriculture': 10, 'Energy': 10},
    # Industries available in this world. Their upper limit.
    'industries': {industry: 5000 for industry in INDUSTRIES},
    'industries_chin': INDUSTRIES_CHIN,
    # (optional) kwargs of the chosen scenario class
    'starting_agent_resources': {"Food": 10., "Energy": 10.}, #food, energy
    'contribution': contribution,
    'contribution_chg_rate': contribution_chg_rate,
    'industry_depreciation': dict(zip(PROVINCES, [industry_weights(INDUSTRIES, all_zeros(INDUSTRIES)) for prvn in PROVINCES])),
    # Involved in reward function.
    'industry_weights': dict(zip(PROVINCES, [industry_weights(INDUSTRIES, all_zeros(INDUSTRIES)) for prvn in PROVINCES])),
    'industry_init_dstr': industry_init_dstr,
    # Use inverse reinforcement learning to know rewaed function of each agent.
    'irl': False,
    'irl_data_path': './data',

    # ===== STANDARD ARGUMENTS ======
    # to be contnued after layout construction on foundation/scenarios/MacroEcon.
    'world_size': [100, 100],
    'n_agents': len(PROVINCES),
    'episode_length': 10, # Number of timesteps per episode
    'multi_action_mode_agents': True,
    'multi_action_mode_planner': True,
    'allow_observation_scaling': False,
    # Upper limit of industries that localGov can build per timestep.
    'flatten_observations': False,
    'flatten_masks': True,
    'dense_log_frequency': 1,
}

env_obj = RLlibEnvWrapper({"env_config_dict": env_config}, verbose=True)

policies = {
    "a": (
        None,  # uses default policy
        env_obj.observation_space,
        env_obj.action_space,
        {}  # define a custom agent policy configuration.
    ),
    "p": (
        None,  # uses default policy
        env_obj.observation_space_pl,
        env_obj.action_space_pl,
        {}  # define a custom planner policy configuration.
    )
}

# In foundation, all the agents have integer ids and the social planner has an id of "p"
policy_mapping_fun = lambda i: "a" if str(i).isdigit() else "p"
policies_to_train = ["a", "p"]

trainer_config = {
    "multiagent": {
        "policies": policies,
        "policies_to_train": policies_to_train,
        "policy_mapping_fn": policy_mapping_fun,
    }
}

trainer_config.update(
    {
        "num_workers": 4,
        "num_envs_per_worker": 2,
        # Other training parameters
        "train_batch_size":  4000,
        "sgd_minibatch_size": 4000,
        "num_gpus": 2,
        "num_gpus_per_worker": 1,
        "num_sgd_iter": 1
    }
)

# We also add the "num_envs_per_worker" parameter for the env. wrapper to index the environments.
env_config = {
    "env_config_dict": env_config,
    "num_envs_per_worker": trainer_config.get('num_envs_per_worker'),   
}

trainer_config.update(
    {
        "env_config": env_config        
    }
)

# Initialize Ray
ray.init(local_mode=True)

# Create the PPO trainer.
trainer = PPOTrainer(
    env=RLlibEnvWrapper,
    config=trainer_config,
    )

# Train for n iterations and report results (mean episode rewards).
# Since we have to guess 10 times and the optimal reward is 0.0
# (exact match between observation and action value),
# we can expect to reach an optimal episode reward of 0.0.
for i in range(3):
    results = trainer.train()
    print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")


# Computing Actions
env = foundation.make_env_instance(**env_config['env_config_dict'])
obs = env.reset()
print("Start Simulation...")
done = {"__all__": False}
actions = {}
while not done["__all__"]:
    for agent_idx in obs.keys():
        agent = env.get_agent(agent_idx)
        if agent_idx == 'p':
            agent_action = trainer.compute_action(obs[agent_idx], policy_id = 'p')
        else:
            agent_action = trainer.compute_action(obs[agent_idx], policy_id = 'a')
        actions[agent_idx] = agent_action
        #actions[agent_idx] = dict(zip(agent._action_names, agent_action))
        #actions = {k: [v for k, v in v.items()] for k, v in actions.items()}
    obs, reward, done, info = env.step(actions)
    actions.clear()

# Get weights of the default local policy
# trainer.get_policy().get_weights()

# Same as above
trainer.workers.local_worker().policy_map["a"].get_weights()
trainer.workers.local_worker().policy_map["p"].get_weights()

# Get list of weights of each worker, including remote replicas
# trainer.workers.foreach_worker(lambda ev: ev.get_policy().get_weights())

# Same as above
# trainer.workers.foreach_worker_with_index(lambda ev, i: ev.get_policy().get_weights())

# Below, we fetch the dense logs for each rollout worker and environment within
dense_logs = {}
# Note: worker 0 is reserved for the trainer actor
for worker in range((trainer_config["num_workers"] > 0), trainer_config["num_workers"] + 1):
    for env_id in range(trainer_config["num_envs_per_worker"]):                                                                 dense_logs["worker={};env_id={}".format(worker, env_id)] = \
        trainer.workers.foreach_worker(lambda w: w.async_env)[worker].envs[env_id].env.previous_episode_dense_log

# We should have num_workers x num_envs_per_worker number of dense logs
"""
import pickle
with open('dense_logs1.pkl', 'wb') as f:
    pickle.dump(dense_logs, f)
print("Save dense_logs1 in dense_logs.pkl")       
"""
### 4b. Generate a dense log from the most recent trainer policy model weights

#We may also use the trainer object directly to play out an episode. The advantage of this approach is that we can re-sample the policy model any number of times and generate several rollouts.

def generate_rollout_from_current_trainer_policy(
    trainer,
    env_obj,
    num_dense_logs=1
):
    dense_logs = {}
    for idx in range(num_dense_logs):
        # Set initial states
        agent_states = {}
        for agent_idx in range(env_obj.env.n_agents):
            agent_states[str(agent_idx)] = trainer.get_policy("a").get_initial_state()
        planner_states = trainer.get_policy("p").get_initial_state()

        # Play out the episode
        obs = env_obj.reset(force_dense_logging=True)
        for t in range(env_obj.env.episode_length):
            actions = {}
            for agent_idx in range(env_obj.env.n_agents):
                # Use the trainer object directly to sample actions for each agent
                actions[str(agent_idx)] = trainer.compute_action(
                    obs[str(agent_idx)],
                    agent_states[str(agent_idx)],
                    policy_id="a",
                    full_fetch=False
                )

            # Action sampling for the planner
            actions["p"] = trainer.compute_action(
                obs['p'],
                planner_states,                                                                                                                    policy_id='p',
                full_fetch=False
            )

            obs, rew, done, info = env_obj.step(actions)
            if done['__all__']:
                break
        dense_logs[idx] = env_obj.env.dense_log
    return dense_logs

dense_logs = generate_rollout_from_current_trainer_policy(
    trainer,
    env_obj,
    num_dense_logs=2
)
"""
with open('dense_logs2.pkl', 'wb') as f:
    pickle.dump(dense_logs, f)
print("Save dense_logs2 in dense_logs.pkl")

diff = compare_difference(dense_logs, env_config["env_config_dict"]["episode_length"])
with open('diff.pkl', 'wb') as f:
    pickle.dump(diff, f)
print("Save diff in dense_logs2.pkl")
"""
dense_logs = simplify_actions(dense_logs)
with open("result.json", "w") as f:
  json.dump(dense_logs, f)
print("Save dense_logs2 in result.json")
diff = compare_difference(dense_logs, env_config["env_config_dict"]["episode_length"])
with open("diff.json", "w") as f2:
  json.dump(diff, f2)
print("Save diff in diff.json")
# plotting utilities for visualizing env. state
#from utils import plotting  
#plotting.plotting(dense_logs)

# Shutdown Ray after use
ray.shutdown()

