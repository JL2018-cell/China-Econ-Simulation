from obtain_data import obtain_data, industry_dstr_over_time
#from obtain_data import industry_dstr_over_time as all_industry_dstr_over_time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import product               # Cartesian product for iterators
import math
import random
import pdb

# allow us to re-use the framework from the src directory
import sys, os
#sys.path.append(os.path.abspath(os.path.join('./irl_maxent')))

from irl_maxent import gridworld as W                       # basic grid-world MDPs
from irl_maxent import trajectory as T                      # trajectory generation
from irl_maxent import optimizer as O                       # stochastic gradient descent optimizer
from irl_maxent import solver as S                          # MDP solver (value-iteration)
from irl_maxent import plot as P                            # helper-functions for plotting

# == Setup and some utility functions ==

def setup_mdp(industry_dstr, buildUpLimit, industries_limit, resourcePt_contribution, INDUSTRIES):
    # create our world
    world = W.IcyGridWorld(industry_dstr, buildUpLimit, industries_limit, resourcePt_contribution, dim = len(INDUSTRIES), size = 2, p_slip = 0.)

    # set up the reward function
    reward = np.zeros(world.n_states)

    # set up terminal states
    terminal = []

    return world, reward, terminal

def align(arr, target_arr):
    assert len(arr) == len(target_arr)
    indices = []
    for elm in target_arr:
        for el in arr:
            if elm in el:
                indices.append(el)
    return indices

def safe_mult(a, b):
    if a == 0. and b == float('inf'):
        return 0.
    else:
        return a * b

def concat_array(arr1, arr2):
    if arr1 is None:
        return arr2
    elif len(arr1.shape) == 1:
        return np.concatenate((arr1[:, None], arr2[:, None]), axis = 1)
    else: #len(arr1.shape) == 2:
        return np.concatenate((arr1, arr2[:, None]), axis = 1)

# == The Actual Algorithm ==

def feature_expectation_from_trajectories(state_point_to_index, features, trajectories):
    n_states, n_features = features.shape

    fe = np.zeros(n_features)

    for t in trajectories:                  # for each trajectory
        for s in t:                # for each state in trajectory
            _s = state_point_to_index(s)
            fe += features[_s, :]            # sum-up features

    return fe / len(trajectories)           # average over trajectories

def initial_probabilities_from_trajectories(state_point_to_index, n_states, trajectories):
    p = np.zeros(n_states)

    for t in trajectories:                  # for each trajectory
        _t = state_point_to_index(t[0])
        p[_t] += 1.0     # increment starting state

    return p / len(trajectories)            # normalize

def compute_expected_svf(p_transition, p_initial, terminal, reward, eps=1e-5):
    n_states, _, n_actions = p_transition.shape
    nonterminal = set(range(n_states)) - set(terminal)  # nonterminal states
    
    # Backward Pass
    # 1. initialize at terminal states
    zs = np.zeros(n_states)                             # zs: state partition function
    zs[terminal] = 1.0

    # k = sampling size
    k = 1
    # 2. perform backward pass
    # longest trajectory: n_states
    for _ in random.sample(range(2 * n_states), k = k):
        # reset action values to zero
        za = np.zeros((n_states, n_actions))            # za: action partition function

        # for each state-action pair
        for s_from, a in product(random.sample(range(n_states), k = k), random.sample(range(n_actions), k = k)):

            # sum over s_to
            for s_to in random.sample(range(n_states), k = k):
                za[s_from, a] += safe_mult(p_transition.p_trans(s_from, s_to, a), np.exp(reward[s_from])) * zs[s_to]
                if np.any(np.isnan(za)):
                    print("Alert! za nan.")
                    pdb.set_trace()
                if np.any(np.isnan(zs)):
                    print("Alert! zs nan.")
                    pdb.set_trace()
        
        # sum over all actions
        zs = za.sum(axis=1)

    # 3. compute local action probabilities
    p_action = za / zs[:, None]
    np.nan_to_num(p_action, 0, 0)
    if np.any(np.isnan(p_action)):
        print("Alert! p_action nan.")
        pdb.set_trace()

    # Forward Pass
    # 4. initialize with starting probability
    d = np.zeros((n_states, 2 * n_states))              # d: state-visitation frequencies
    d[:, 0] = p_initial

    # 5. iterate for N steps
    for t in random.sample(range(1, 2 * n_states), k = k):                    # longest trajectory: n_states
        
        # for all states
        for s_to in random.sample(range(n_states), k = k):
            
            # sum over nonterminal state-action pairs
            for s_from, a in product(random.sample(nonterminal, k = k), random.sample(range(n_actions), k = k)):
                d[s_to, t] += d[s_from, t-1] * p_action[s_from, a] * p_transition.p_trans(s_from, s_to, a)

    if np.any(np.isnan(d)):
        print("Alert! d nan.")
        pdb.set_trace()

    # 6. sum-up frequencies
    return d.sum(axis=1)

def maxent_irl(world, p_transition, features, terminal, trajectories, optim, init, eps=1e-4):
    n_states, _, n_actions = p_transition.shape
    _, n_features = features.shape

    # compute feature expectation from trajectories
    e_features = feature_expectation_from_trajectories(world.state_point_to_index, features, trajectories)
    
    # compute starting-state probabilities from trajectories
    p_initial = initial_probabilities_from_trajectories(world.state_point_to_index, n_states, trajectories)

    # gradient descent optimization
    omega = init(n_features)        # initialize our parameters
    delta = np.inf                  # initialize delta for convergence check

    optim.reset(omega)              # re-start optimizer
    while delta > eps:              # iterate until convergence
        print(delta)

        omega_old = omega.copy()

        # compute per-state reward from features
        reward = features.dot(omega)

        # compute gradient of the log-likelihood
        e_svf = compute_expected_svf(p_transition, p_initial, terminal, reward)
        grad = e_features - features.T.dot(e_svf)

        # perform optimization step and compute delta for convergence
        optim.step(-grad)
        
        # re-compute detla for convergence check
        delta = np.max(np.abs(omega_old - omega))

    # re-compute per-state reward and return
    return features.dot(omega)

# == Main Program ==

def irl_maxent(DATA_PATH, provinces, INDUSTRIES, INDUSTRIES_CHIN, industry_init_dstr, contribution, buildUpLimit, industries_limit):
    from obtain_data import industry_dstr_over_time
    #DATA_PATH = './data'
    #industry_init_dstr, contribution = obtain_data(DATA_PATH)
    #INDUSTRIES_CHIN = ["农林牧渔业", "电力、热力、燃气及水生产和供应业", "金融业", "信息传输、软件和信息技术服务业",  "采矿业", "住宿和餐饮业", "制造业", "建筑业", "交通运输、仓储和邮政业", "批发和零售业", "教育业"]
    #INDUSTRIES = ['Agriculture', 'Energy', 'Finance', 'IT', 'Minerals', 'Tourism', 'Manufacturing', 'Construction', 'Transport', 'Retail', 'Education']
    #provinces = list(industry_init_dstr.keys())
    #buildUpLimit = {'Agriculture': 10, 'Energy': 10}
    # Industries available in this world. Their upper limit.
    #industries_limit = {industry: 2000 for industry in INDUSTRIES}
    # Solution of reward function of each agent
    reward_matrices = {}

    for province in provinces: 
        print("Initializing...")
        # set-up the GridWorld Markov Decision Process
        world, reward, terminal = setup_mdp(industry_init_dstr[province], buildUpLimit, industries_limit, contribution["resource_points"][province], INDUSTRIES)
        industry_dstr_ovr_t = industry_dstr_over_time(DATA_PATH)[province]
        
        # Read data, convert ot trajectories.
        indices = [[col for col in industry_dstr_ovr_t.columns if industry in col][0] for industry in INDUSTRIES_CHIN]
        indices = align(indices, INDUSTRIES_CHIN)
        industry_dstr_ovr_t = industry_dstr_ovr_t[indices]
        trajectories = [industry_dstr_ovr_t.to_numpy()]
        
            
        # Convert data to binary grids.
        max_vals = [np.max(trajectory) for trajectory in trajectories]
        reward_maxent = None
        # Iterate over trajectories.
        for max_val in max_vals:
            tmp = max_val // 2 + 1
            while tmp > 1:
                trajectories = [(trajectory // tmp).astype('int32') for i, trajectory in enumerate(trajectories)]
            
                print("Set up features.")
                # set up features: we use one feature vector per state
                features = W.state_features(world)
                
                print("Choose parameters.")
                
                # choose our parameter initialization strategy:
                #   initialize parameters with constant
                init = O.Constant(1.0)
                
                print("Optimizing...")
                
                # choose our optimization strategy:
                #   we select exponentiated stochastic gradient descent with linear learning-rate decay
                optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))
                
                print("RL learning...")
                
                # actually do some inverse reinforcement learning
                reward_maxent = concat_array(reward_maxent, maxent_irl(world, world.p_transition, features, terminal, trajectories, optim, init))
                
                print("Done!")
                print(reward_maxent)
                tmp = tmp // 2

        reward_matrices[province] = reward_maxent
        #np.savetxt(province + ".csv", reward_maxent, delimiter=",")
    return reward_matrices
    
