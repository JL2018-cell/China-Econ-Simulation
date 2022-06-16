import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from collections import namedtuple


import img_utils
from mdp import gridworld
from mdp import value_iteration
from deep_maxent_irl import *
from maxent_irl import *
from utils import *
from lp_irl import *
import foundation

Step = namedtuple('Step','cur_state action next_state reward done')


PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-hei', '--height', default=5, type=int, help='height of the gridworld')
PARSER.add_argument('-wid', '--width', default=5, type=int, help='width of the gridworld')
PARSER.add_argument('-g', '--gamma', default=0.9, type=float, help='discount factor')
PARSER.add_argument('-a', '--act_random', default=0.3, type=float, help='probability of acting randomly')
PARSER.add_argument('-t', '--n_trajs', default=200, type=int, help='number of expert trajectories')
PARSER.add_argument('-l', '--l_traj', default=20, type=int, help='length of expert trajectory')
PARSER.add_argument('--rand_start', dest='rand_start', action='store_true', help='when sampling trajectories, randomly pick start positions')
PARSER.add_argument('--no-rand_start', dest='rand_start',action='store_false', help='when sampling trajectories, fix start positions')
PARSER.set_defaults(rand_start=True)
PARSER.add_argument('-lr', '--learning_rate', default=0.02, type=float, help='learning rate')
PARSER.add_argument('-ni', '--n_iters', default=20, type=int, help='number of iterations')
PARSER.add_argument('-d', '--data_src', default="Data.xlsx", type=str, help='data source path')
ARGS = PARSER.parse_args()
print("ARGS:", ARGS)


GAMMA = ARGS.gamma
ACT_RAND = ARGS.act_random
R_MAX = 1 # the constant r_max does not affect much the recoverred reward distribution
H = ARGS.height
W = ARGS.width
N_TRAJS = ARGS.n_trajs
L_TRAJ = ARGS.l_traj
RAND_START = ARGS.rand_start
LEARNING_RATE = ARGS.learning_rate
N_ITERS = ARGS.n_iters
DATA_SRC = ARGS.data_src
PROVINCES = ["GuangDong", "HeBei", "XinJiang"]

def generate_demonstrations():
    gdp = pd.read_excel(DATA_SRC, sheet_name = 0, header = 0, index_col = 0)
    gdp = gdp.fillna(0)
    #gdp = gdp.apply(lambda x: x/reduce_scale)
    #gdp = gdp.round(decimals = 0)
    #gdp = gdp.astype('int32')
    pollutant = pd.read_excel(DATA_SRC, sheet_name = 1, header = 0, index_col = 0)
    pollutant = pollutant.fillna(0)
    #pollutant = pollutant.apply(lambda x: x/reduce_scale)
    #pollutant = pollutant.round(decimals = 0)
    #pollutant = pollutant.astype('int32')
    industry = pd.read_excel(DATA_SRC, sheet_name = 2, header = 0, index_col = 0)
    industry = industry.fillna(0)
    #industry = industry.apply(lambda x: x/reduce_scale)
    #industry = industry.round(decimals = 0)
    #industry = industry.astype('int32')

    tjs = {}
    max = 0
    base_yr = min(industry.columns)

    # format:
    # {province name: array([[state 1], [state 2], [state 3], ... [final state]])}
    # 1 trajectory for each province
    # Example: Step(cur_state = [1, 2, 3], action = [0, 0, 0], next_state = [1, 2, 3], reward = 0, done = False)
    for prvn in PROVINCES:
        tjs[prvn] = []
        hist = industry.loc[[idx for idx in industry.index if prvn in idx]].T.to_numpy()
        print("Max:", hist.max())
        max = hist.max() if hist.max() > max else max
        for i in range(hist.shape[0] - 1):
           reward = gdp.loc[prvn, base_yr + i] - pollutant.loc[[idx for idx in pollutant.index if "GuangDong" in idx]].sum()[base_yr + i]
           # [(state, action, next_state)]
           tjs[prvn].append(Step(cur_state = hist[i], action = hist[i + 1] - hist[i], next_state = hist[i + 1], reward = reward, done = False))

    return tjs, max


def main():


  # replace with MacroEcon layout env
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
      'industries': {'Agriculture': 2000, 'Energy': 2000, 'Finance': 2000, \
                                 'IT': 2000, 'Minerals': 2000, 'Tourism': 2000}, #Help to define actions of localGov
  
      # (optional) kwargs of the chosen scenario class
      'starting_agent_resources': {"Food": 10., "Energy": 10.}, #food, energy
      'industry_depreciation': {"GuangDong": {'Agriculture': 1, 'Energy': 1, 'Finance': 1, 'IT': 1, 'Minerals': 1, 'Tourism': 1}, "HeBei": {'Agriculture': 1, 'Energy': 1, 'Finance': 1, 'IT': 1, 'Minerals': 1, 'Tourism': 1}, "XinJiang": {'Agriculture': 1, 'Energy': 1, 'Finance': 1, 'IT': 1, 'Minerals': 1, 'Tourism': 1}}, #Help to calculate rewards.
      'industry_init_dstr': {"GuangDong": {'Agriculture': 1, 'Energy': 1, 'Finance': 1, 'IT': 1, 'Minerals': 1, 'Tourism': 1}, "HeBei": {'Agriculture': 1, 'Energy': 1, 'Finance': 1, 'IT': 1, 'Minerals': 1, 'Tourism': 1}, "XinJiang": {'Agriculture': 1, 'Energy': 1, 'Finance': 1, 'IT': 1, 'Minerals': 1, 'Tourism': 1}}, #Help to calculate rewards.
      'industry_weights': {"GuangDong": {'Agriculture': 1., 'Energy': 1., 'Finance': 1., 'IT': 1., 'Minerals': 1., 'Tourism': 1.}, "HeBei": {'Agriculture': 1., 'Energy': 1., 'Finance': 1., 'IT': 1., 'Minerals': 1., 'Tourism': 1.}, "XinJiang": {'Agriculture': 1., 'Energy': 1., 'Finance': 1., 'IT': 1., 'Minerals': 1., 'Tourism': 1.}}, #Help to calculate rewards.
      'dense_log_frequency': 1
  }
  gw = foundation.make_env_instance(**env_config)
  obs = gw.reset()

  # Generate expert trajectories
  # Example: Step(cur_state = [1, 2, 3], action = [0, 0, 0], next_state = [1, 2, 3], reward = 0, done = False)
  trajs, max = generate_demonstrations()

  # Use neural network to remeber reward
  # Or use lambda function
  P_a = gw.get_transition_pr(0)

  # use identity matrix as feature
  N_STATES = H * W
  feat_map = np.eye(N_STATES)
  
  print('Deep Max Ent IRL training ..')
  rewards = deep_maxent_irl(gw.world.agents[0].state["inventory"], feat_map, P_a, GAMMA, trajs, LEARNING_RATE, N_ITERS)

  values, _ = value_iteration.value_iteration(P_a, rewards, GAMMA, error=0.01, deterministic=True)



if __name__ == "__main__":
  main()
