import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import product               # Cartesian product for iterators

# allow us to re-use the framework from the src directory
import sys, os
sys.path.append(os.path.abspath(os.path.join('../src/irl_maxent')))

import gridworld as W                       # basic grid-world MDPs
import trajectory as T                      # trajectory generation
import optimizer as O                       # stochastic gradient descent optimizer
import solver as S                          # MDP solver (value-iteration)
import plot as P                            # helper-functions for plotting

PROVINCES = ["GuangDong", "HeBei", "XinJiang"]

def setup_mdp():
    # create our world
    world = W.GridWorld()

    # set up the reward function
    CO2 = {'Agriculture': 2000, 'Energy': 2000, 'Finance': 2000, \
           'IT': 2000, 'Minerals': 2000, 'Tourism': 2000}
    GDP = {'Agriculture': 2000, 'Energy': 2000, 'Finance': 2000, \
           'IT': 2000, 'Minerals': 2000, 'Tourism': 2000}
    reward = lambda s: np.dot(np.array(GDP.values), s) - np.dot(np.array(CO2.values), s)

    # No terminal state
    terminal = []

    return world, reward, terminal

# set-up the GridWorld Markov Decision Process
world, reward, terminal = setup_mdp()

# generate some "expert" trajectories from data.
def generate_expert_trajectories():
    gdp = pd.read_excel('Data.xlsx', sheet_name = 0, header = 0, index_col = 0)
    gdp = gdp.fillna(0)
    gdp = gdp.apply(lambda x: x * 10)
    gdp = gdp.astype('int32')
    pollutant = pd.read_excel('Data.xlsx', sheet_name = 1, header = 0, index_col = 0)
    pollutant = pollutant.fillna(0)
    pollutant = pollutant.apply(lambda x: x * 10)
    pollutant = pollutant.astype('int32')
    industry = pd.read_excel('Data.xlsx', sheet_name = 2, header = 0, index_col = 0)
    industry = industry.fillna(0)
    industry = industry.apply(lambda x: x * 10)
    industry = industry.astype('int32')

    tjs = {}

    # format:
    # {province name: array([[state 1], [state 2], [state 3], ... [final state]])}
    # 1 trajectory for each province
    for prvn in PROVINCES:
        tjs[prvn] = []
        hist = industry.loc[[idx for idx in industry.index if prvn in idx]].T.to_numpy()
        print("Max:", hist.max())
        for i in range(hist.shape[0] - 1):
           # [(state, action, next_state)]
           tjs[prvn].append((hist[i], hist[i + 1] - hist[i], hist[i + 1]))
        tjs[prvn] = [T.Trajectory(tjs[prvn])]

    policy = []
    return tjs

# generate some "expert" trajectories (and its policy for visualization)
trajectories = generate_expert_trajectories()

# == The Actual Algorithm ==

#Need number of features
def feature_expectation_from_trajectories(industries, n_states, n_features, features, trajectories):

    fe = np.zeros(features.shape)

    for t in trajectories:                  # for each trajectory
        for s in t.states():                # for each state in trajectory
            for _s in s:                    # for each item of state
                print("_s", _s)
                fe += np.array([np.concatenate((np.ones(_s), np.zeros(v - _s)), axis = 0) \
                                               for v in industries.values()]).flatten() # sum up features
    return fe / len(trajectories)           # average over trajectories

def initial_probabilities_from_trajectories(n_states, trajectories):
    p = np.zeros(n_states)

    for t in trajectories:                  # for each trajectory
        p[t.transitions()[0][0]] += 1.0     # increment starting state

    return p / len(trajectories)            # normalize


def compute_expected_svf(n_states, n_actions, p_transition, p_initial, terminal, reward, eps=1e-5):
    nonterminal = set(range(n_states)) - set(terminal)  # nonterminal states
    
    # Backward Pass
    # 1. initialize at terminal states
    zs = np.zeros(n_states)                             # zs: state partition function
    zs[terminal] = 1.0

    # 2. perform backward pass
    for _ in range(2 * n_states):                       # longest trajectory: n_states
        # reset action values to zero
        za = np.zeros((n_states, n_actions))            # za: action partition function

        # for each state-action pair
        for s_from, a in product(range(n_states), range(n_actions)):

            # sum over s_to
            for s_to in range(n_states):
                za[s_from, a] += p_transition[s_from, s_to, a] * np.exp(reward[s_from]) * zs[s_to]
        
        # sum over all actions
        zs = za.sum(axis=1)

    # 3. compute local action probabilities
    p_action = za / zs[:, None]

    # Forward Pass
    # 4. initialize with starting probability
    d = np.zeros((n_states, 2 * n_states))              # d: state-visitation frequencies
    d[:, 0] = p_initial

    # 5. iterate for N steps
    for t in range(1, 2 * n_states):                    # longest trajectory: n_states
        
        # for all states
        for s_to in range(n_states):
            
            # sum over nonterminal state-action pairs
            for s_from, a in product(nonterminal, range(n_actions)):
                d[s_to, t] += d[s_from, t-1] * p_action[s_from, a] * p_transition[s_from, s_to, a]

    # 6. sum-up frequencies
    return d.sum(axis=1)

def maxent_irl(n_states, n_actions, p_transition, features, terminal, trajectories, optim, init, eps=1e-4):
    # Refer to self.n_actions, self.n_states defined in gridworld.py
    n_features = n_states

    # compute feature expectation from trajectories
    e_features = feature_expectation_from_trajectories(world.industries, n_states, n_features, features, trajectories)
    
    # compute starting-state probabilities from trajectories
    p_initial = initial_probabilities_from_trajectories(n_states, trajectories)

    # gradient descent optimization
    omega = init(n_features)        # initialize our parameters
    delta = np.inf                  # initialize delta for convergence check

    optim.reset(omega)              # re-start optimizer
    while delta > eps:              # iterate until convergence
        omega_old = omega.copy()

        # compute per-state reward from features
        reward = features.dot(omega)

        # compute gradient of the log-likelihood
        e_svf = compute_expected_svf(n_states, n_actions, p_transition, p_initial, terminal, reward)
        grad = e_features - features.T.dot(e_svf)

        # perform optimization step and compute delta for convergence
        optim.step(grad)
        
        # re-compute detla for convergence check
        delta = np.max(np.abs(omega_old - omega))

    # re-compute per-state reward and return
    return features.dot(omega)

print("Set up features.")

# set up features: we use >1 feature vector per state
features = W.state_features(world)

print("Choose parameters.")

# choose our parameter initialization strategy:
#   initialize parameters with constant
init = O.Constant(1.0)

print("Optimizing...")

# choose our optimization strategy:
#   we select exponentiated stochastic gradient descent with linear learning-rate decay
optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

print("Reverse RL learning...")

# actually do some inverse reinforcement learning
reward_maxent = {}
for prvn in PROVINCES:
    reward_maxent[prvn] = maxent_irl(world.n_states, world.n_actions, world.p_transitions, features, terminal, trajectories[prvn], optim, init)
    print("Finish learning data of", prvn)

pd.DataFrame(reward_maxent).to_pickle("./reward_maxent.pkl", compression='infer', protocol=5, storage_options=None)
print("Done!")


"""
fig = plt.figure()
ax = fig.add_subplot(121)
ax.title.set_text('Original Reward')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
p = P.plot_state_values(ax, world, reward, **style)
P.plot_deterministic_policy(ax, world, S.optimal_policy(world, reward, 0.8), color='red')
fig.colorbar(p, cax=cax)

ax = fig.add_subplot(122)
ax.title.set_text('Recovered Reward')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
p = P.plot_state_values(ax, world, reward_maxent, **style)
P.plot_deterministic_policy(ax, world, S.optimal_policy(world, reward_maxent, 0.8), color='red')
fig.colorbar(p, cax=cax)

fig.tight_layout()
#plt.show()
plt.savefig('plot2.png', dpi=250)

# Note: this code will only work with one feature per state
p_initial = initial_probabilities_from_trajectories(world.n_states, trajectories)
e_svf = compute_expected_svf(world.p_transition, p_initial, terminal, reward_maxent)
e_features = feature_expectation_from_trajectories(features, trajectories)

fig = plt.figure()
ax = fig.add_subplot(121)
ax.title.set_text('Trajectory Feature Expectation')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
p = P.plot_state_values(ax, world, e_features, **style)
fig.colorbar(p, cax=cax)

ax = fig.add_subplot(122)
ax.title.set_text('MaxEnt Feature Expectation')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
p = P.plot_state_values(ax, world, features.T.dot(e_svf), **style)
fig.colorbar(p, cax=cax)

fig.tight_layout()
#plt.show()
plt.savefig('plot3.png', dpi=250)

"""

