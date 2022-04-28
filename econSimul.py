from ray.rllib.agents.ppo import PPOTrainer
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


# Create an RLlib Trainer instance to learn how to act in the above
# environment.
trainer = PPOTrainer(
    config={
        # Env class to use (here: our gym.Env sub-class from above).
        "env": MacroEconLayout,
        # Config dict to be passed to our custom env's constructor.
        "env_config": env_config,
        # Parallelize environment rollouts.
        "num_workers": 2,
    })


# Train for n iterations and report results (mean episode rewards).
# Since we have to guess 10 times and the optimal reward is 0.0
# (exact match between observation and action value),
# we can expect to reach an optimal episode reward of 0.0.
for i in range(1):
    results = trainer.train()
    print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")

